#!/usr/bin/env python3
"""
Enhanced Hybrid Reference Checker with Multiple API Sources

This module provides an improved hybrid reference checker that intelligently combines
multiple API sources for optimal reliability and performance. It replaces Google Scholar
with more reliable alternatives while maintaining backward compatibility.

New API Integration Priority:
1. Local Semantic Scholar Database (fastest, offline)
2. Semantic Scholar API (reliable, good coverage)  
3. OpenAlex API (excellent reliability, replaces Google Scholar)
4. CrossRef API (best for DOI-based verification)
5. Google Scholar (final fallback, kept for legacy support)

Usage:
    from enhanced_hybrid_checker import EnhancedHybridReferenceChecker
    
    checker = EnhancedHybridReferenceChecker(
        semantic_scholar_api_key="your_key",
        db_path="path/to/db.sqlite",
        contact_email="your@email.com"
    )
    
    verified_data, errors, url = checker.verify_reference(reference)
"""

import logging
import random
import requests
import time
from typing import Dict, List, Tuple, Optional, Any

logger = logging.getLogger(__name__)

class EnhancedHybridReferenceChecker:
    """
    Enhanced hybrid reference checker with multiple API sources for improved reliability
    """
    
    def __init__(self, semantic_scholar_api_key: Optional[str] = None, 
                 db_path: Optional[str] = None,
                 contact_email: Optional[str] = None,
                 enable_openalex: bool = True,
                 enable_crossref: bool = True,
                 debug_mode: bool = False):
        """
        Initialize the enhanced hybrid reference checker
        
        Args:
            semantic_scholar_api_key: Optional API key for Semantic Scholar
            db_path: Optional path to local Semantic Scholar database
            contact_email: Email for polite pool access to APIs
            enable_openalex: Whether to use OpenAlex API
            enable_crossref: Whether to use CrossRef API
            debug_mode: Whether to enable debug logging
        """
        self.contact_email = contact_email
        self.debug_mode = debug_mode
        
        # Initialize local database checker if available
        self.local_db = None
        if db_path:
            try:
                from .local_semantic_scholar import LocalNonArxivReferenceChecker
                self.local_db = LocalNonArxivReferenceChecker(db_path=db_path)
                logger.debug(f"Enhanced Hybrid: Local database enabled at {db_path}")
            except Exception as e:
                logger.warning(f"Enhanced Hybrid: Failed to initialize local database: {e}")
                self.local_db = None
        
        # Initialize Semantic Scholar API
        try:
            from .semantic_scholar import NonArxivReferenceChecker
            self.semantic_scholar = NonArxivReferenceChecker(api_key=semantic_scholar_api_key)
            logger.debug("Enhanced Hybrid: Semantic Scholar API initialized")
        except Exception as e:
            logger.error(f"Enhanced Hybrid: Failed to initialize Semantic Scholar: {e}")
            self.semantic_scholar = None
        
        # Initialize OpenAlex API
        self.openalex = None
        if enable_openalex:
            try:
                from .openalex import OpenAlexReferenceChecker
                self.openalex = OpenAlexReferenceChecker(email=contact_email)
                logger.debug("Enhanced Hybrid: OpenAlex API initialized")
            except Exception as e:
                logger.warning(f"Enhanced Hybrid: Failed to initialize OpenAlex: {e}")
        
        # Initialize CrossRef API
        self.crossref = None
        if enable_crossref:
            try:
                from .crossref import CrossRefReferenceChecker
                self.crossref = CrossRefReferenceChecker(email=contact_email)
                logger.debug("Enhanced Hybrid: CrossRef API initialized")
            except Exception as e:
                logger.warning(f"Enhanced Hybrid: Failed to initialize CrossRef: {e}")
        
        # Initialize OpenReview checker
        self.openreview = None
        try:
            from .openreview_checker import OpenReviewReferenceChecker
            self.openreview = OpenReviewReferenceChecker()
            logger.debug("Enhanced Hybrid: OpenReview checker initialized")
        except Exception as e:
            logger.warning(f"Enhanced Hybrid: Failed to initialize OpenReview: {e}")
            self.openreview = None
        
        # Google Scholar removed - using more reliable APIs only
        
        # Track API performance for adaptive selection
        self.api_stats = {
            'local_db': {'success': 0, 'failure': 0, 'avg_time': 0, 'throttled': 0},
            'semantic_scholar': {'success': 0, 'failure': 0, 'avg_time': 0, 'throttled': 0},
            'openalex': {'success': 0, 'failure': 0, 'avg_time': 0, 'throttled': 0},
            'crossref': {'success': 0, 'failure': 0, 'avg_time': 0, 'throttled': 0},
            'openreview': {'success': 0, 'failure': 0, 'avg_time': 0, 'throttled': 0}
        }
        
        # Track failed API calls for retry logic - OPTIMIZED CONFIGURATION
        self.retry_base_delay = 1  # Base delay for retrying throttled APIs (seconds)
        self.retry_backoff_factor = 1.5  # Exponential backoff multiplier
        self.max_retry_delay = 20  # Maximum delay cap in seconds
    
    def _update_api_stats(self, api_name: str, success: bool, duration: float):
        """Update API performance statistics"""
        if api_name in self.api_stats:
            stats = self.api_stats[api_name]
            if success:
                stats['success'] += 1
            else:
                stats['failure'] += 1
            
            # Update average time (simple moving average)
            total_calls = stats['success'] + stats['failure']
            stats['avg_time'] = ((stats['avg_time'] * (total_calls - 1)) + duration) / total_calls
    
    def _try_api(self, api_name: str, api_instance: Any, reference: Dict[str, Any], is_retry: bool = False) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]], Optional[str], bool, str]:
        """
        Try to verify reference with a specific API and track performance
        
        Returns:
            Tuple of (verified_data, errors, url, success, failure_type)
            failure_type can be: 'none', 'not_found', 'throttled', 'timeout', 'other'
        """
        if not api_instance:
            return None, [], None, False, 'none'
        
        start_time = time.time()
        failure_type = 'none'
        
        try:
            verified_data, errors, url = api_instance.verify_reference(reference)
            duration = time.time() - start_time
            
            # Check if we got API failure errors indicating retryable failure
            api_failure_errors = [err for err in errors if err.get('error_type') == 'api_failure']
            if api_failure_errors:
                # This is a retryable API failure, not a verification result
                self._update_api_stats(api_name, False, duration)
                logger.debug(f"Enhanced Hybrid: {api_name} API failed in {duration:.2f}s: {api_failure_errors[0].get('error_details', 'unknown')}")
                return None, [], None, False, 'throttled'  # Treat API failures as throttling for retry logic
            
            # Consider it successful if we found data or verification errors (i.e., we could verify something)
            success = verified_data is not None or len(errors) > 0
            self._update_api_stats(api_name, success, duration)
            
            if success:
                retry_info = " (retry)" if is_retry else ""
                logger.debug(f"Enhanced Hybrid: {api_name} successful in {duration:.2f}s{retry_info}, URL: {url}")
                return verified_data, errors, url, True, 'none'
            else:
                logger.debug(f"Enhanced Hybrid: {api_name} found no results in {duration:.2f}s")
                return None, [], None, False, 'not_found'
                
        except requests.exceptions.Timeout as e:
            duration = time.time() - start_time
            self._update_api_stats(api_name, False, duration)
            failure_type = 'timeout'
            print(f"      ⏱️ TIMEOUT: {api_name} timed out after {duration:.2f}s")
            logger.debug(f"Enhanced Hybrid: {api_name} timed out in {duration:.2f}s: {e}")
            return None, [], None, False, failure_type
            
        except requests.exceptions.RequestException as e:
            duration = time.time() - start_time
            self._update_api_stats(api_name, False, duration)
            
            # Check if it's a rate limiting or server error that should be retried
            error_str = str(e).lower()
            status_code = getattr(e.response, 'status_code', None) if hasattr(e, 'response') and e.response else None
            
            if (status_code == 429) or "429" in str(e) or "rate limit" in error_str:
                failure_type = 'throttled'
                self.api_stats[api_name]['throttled'] += 1
                print(f"      ⚠️ RATE LIMITED (429): {api_name} throttled after {duration:.2f}s")
                logger.debug(f"Enhanced Hybrid: {api_name} rate limited in {duration:.2f}s: {e}")
            elif (status_code and status_code >= 500) or "500" in str(e) or "502" in str(e) or "503" in str(e) or "server error" in error_str or "service unavailable" in error_str:
                failure_type = 'server_error'
                print(f"      🔥 SERVER ERROR ({status_code}): {api_name} failed after {duration:.2f}s")
                logger.debug(f"Enhanced Hybrid: {api_name} server error in {duration:.2f}s: {e}")
            else:
                failure_type = 'other'
                logger.debug(f"Enhanced Hybrid: {api_name} failed in {duration:.2f}s: {e}")
            return None, [], None, False, failure_type
            
        except Exception as e:
            duration = time.time() - start_time
            self._update_api_stats(api_name, False, duration)
            failure_type = 'other'
            logger.debug(f"Enhanced Hybrid: {api_name} failed in {duration:.2f}s: {e}")
            return None, [], None, False, failure_type
    
    def _should_try_doi_apis_first(self, reference: Dict[str, Any]) -> bool:
        """
        Determine if we should prioritize DOI-based APIs (CrossRef) for this reference
        """
        # Check if reference has DOI information
        has_doi = (reference.get('doi') or 
                  (reference.get('url') and ('doi.org' in reference['url'] or 'doi:' in reference['url'])) or
                  (reference.get('raw_text') and ('doi' in reference['raw_text'].lower())))
        return has_doi
    
    def _is_data_complete(self, verified_data: Dict[str, Any], reference: Dict[str, Any]) -> bool:
        """
        Check if the verified data is sufficiently complete for the reference verification
        
        Args:
            verified_data: Paper data returned by API
            reference: Original reference data
            
        Returns:
            True if data is complete enough to use, False if incomplete
        """
        if not verified_data:
            return False
        
        # If the reference has authors, the verified data should also have authors
        cited_authors = reference.get('authors', [])
        found_authors = verified_data.get('authors', [])
        
        # If we cited authors but found none, the data is incomplete
        if cited_authors and not found_authors:
            logger.debug(f"Enhanced Hybrid: Data incomplete - cited authors {cited_authors} but found none")
            return False
        
        return True
    
    def verify_reference(self, reference: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]], Optional[str]]:
        """
        Verify a non-arXiv reference using multiple APIs in priority order
        
        First tries all APIs once, then retries failed APIs if no success.
        
        Args:
            reference: Reference data dictionary
            
        Returns:
            Tuple of (verified_data, errors, url)
        """
        # Track all APIs that failed and could be retried
        failed_apis = []
        
        # PHASE 1: Try all APIs once in priority order
        
        # Strategy 1: Always try local database first (fastest)
        if self.local_db:
            verified_data, errors, url, success, failure_type = self._try_api('local_db', self.local_db, reference)
            if success:
                return verified_data, errors, url
            if failure_type in ['throttled', 'timeout', 'server_error']:
                failed_apis.append(('local_db', self.local_db, failure_type))
        
        # Strategy 2: If reference has DOI, prioritize CrossRef
        crossref_result = None
        if self._should_try_doi_apis_first(reference) and self.crossref:
            verified_data, errors, url, success, failure_type = self._try_api('crossref', self.crossref, reference)
            if success:
                # Check if the data is complete enough to use
                if self._is_data_complete(verified_data, reference):
                    return verified_data, errors, url
                else:
                    # Data is incomplete, save it as fallback and continue with other APIs
                    crossref_result = (verified_data, errors, url)
                    logger.debug("Enhanced Hybrid: CrossRef data incomplete, continuing with other APIs")
            if failure_type in ['throttled', 'timeout', 'server_error']:
                failed_apis.append(('crossref', self.crossref, failure_type))
        
        # Strategy 3: Try Semantic Scholar API (reliable, good coverage)
        if self.semantic_scholar:
            verified_data, errors, url, success, failure_type = self._try_api('semantic_scholar', self.semantic_scholar, reference)
            if success:
                return verified_data, errors, url
            # For Semantic Scholar, only retry retryable failures (not 'not_found')
            if failure_type in ['throttled', 'timeout', 'server_error']:
                failed_apis.append(('semantic_scholar', self.semantic_scholar, failure_type))
        
        # Strategy 4: Try OpenAlex API (excellent reliability, replaces Google Scholar)
        openalex_result = None
        if self.openalex:
            verified_data, errors, url, success, failure_type = self._try_api('openalex', self.openalex, reference)
            if success:
                # Check if the data is complete enough to use
                if self._is_data_complete(verified_data, reference):
                    return verified_data, errors, url
                else:
                    # Data is incomplete, save it as fallback and continue with other APIs
                    openalex_result = (verified_data, errors, url)
                    logger.debug("Enhanced Hybrid: OpenAlex data incomplete, continuing with other APIs")
            if failure_type in ['throttled', 'timeout', 'server_error']:
                failed_apis.append(('openalex', self.openalex, failure_type))
        
        # Strategy 5: Try OpenReview if URL suggests it's an OpenReview paper
        if (self.openreview and 
            hasattr(self.openreview, 'is_openreview_reference') and 
            self.openreview.is_openreview_reference(reference)):
            logger.debug("Enhanced Hybrid: Trying OpenReview URL-based verification")
            verified_data, errors, url, success, failure_type = self._try_api('openreview', self.openreview, reference)
            if success:
                return verified_data, errors, url
            if failure_type in ['throttled', 'timeout', 'server_error']:
                failed_apis.append(('openreview', self.openreview, failure_type))
        
        # Strategy 5b: Try OpenReview by search if venue suggests it might be there
        elif (self.openreview and 
              hasattr(self.openreview, 'verify_reference_by_search')):
            # Check if venue suggests this might be on OpenReview
            venue = reference.get('venue', reference.get('journal', '')).lower()
            openreview_venues = [
                'iclr', 'icml', 'neurips', 'nips', 'aaai', 'ijcai', 
                'international conference on learning representations',
                'international conference on machine learning',
                'neural information processing systems'
            ]
            
            venue_suggests_openreview = any(or_venue in venue for or_venue in openreview_venues)
            logger.debug(f"Enhanced Hybrid: OpenReview venue check - venue: '{venue}', suggests: {venue_suggests_openreview}")
            
            if venue_suggests_openreview:
                logger.debug("Enhanced Hybrid: Trying OpenReview search-based verification")
                verified_data, errors, url, success, failure_type = self._try_openreview_search(reference)
                if success:
                    return verified_data, errors, url
                if failure_type in ['throttled', 'timeout', 'server_error']:
                    failed_apis.append(('openreview_search', self.openreview, failure_type))
        
        # Strategy 6: Try CrossRef if we haven't already (for non-DOI references)
        if not self._should_try_doi_apis_first(reference) and self.crossref:
            verified_data, errors, url, success, failure_type = self._try_api('crossref', self.crossref, reference)
            if success:
                # Check if the data is complete enough to use
                if self._is_data_complete(verified_data, reference):
                    return verified_data, errors, url
                else:
                    # Data is incomplete, save it as fallback
                    if not crossref_result:  # Only save if we don't already have one
                        crossref_result = (verified_data, errors, url)
                        logger.debug("Enhanced Hybrid: CrossRef data incomplete (non-DOI), continuing with other APIs")
            if failure_type in ['throttled', 'timeout', 'server_error']:
                failed_apis.append(('crossref', self.crossref, failure_type))
        
        # PHASE 2: If no API succeeded in Phase 1, retry failed APIs
        if failed_apis:
            logger.debug(f"Enhanced Hybrid: Phase 1 complete, no success. Retrying {len(failed_apis)} failed APIs")
            
            # Sort failed APIs to prioritize Semantic Scholar retries
            semantic_scholar_retries = [api for api in failed_apis if api[0] == 'semantic_scholar']
            other_retries = [api for api in failed_apis if api[0] != 'semantic_scholar']
            
            # Try other APIs first, then Semantic Scholar with more aggressive retries
            retry_order = other_retries + semantic_scholar_retries
            
            for api_name, api_instance, failure_type in retry_order:
                # Use base delay for first retry of each API
                delay = min(self.retry_base_delay, self.max_retry_delay)
                
                # Add jitter to prevent thundering herd (±25% randomization)
                jitter = delay * 0.25 * (2 * random.random() - 1)
                final_delay = max(0.5, delay + jitter)
                
                print(f"      🔄 RETRY: Waiting {final_delay:.1f}s before retrying {api_name} ({failure_type})")
                logger.debug(f"Enhanced Hybrid: Waiting {final_delay:.1f}s before retrying {api_name} after {failure_type} failure")
                time.sleep(final_delay)
                
                logger.debug(f"Enhanced Hybrid: Retrying {api_name}")
                verified_data, errors, url, success, _ = self._try_api(api_name, api_instance, reference, is_retry=True)
                if success:
                    logger.debug(f"Enhanced Hybrid: {api_name} succeeded on retry after {failure_type} (delay: {final_delay:.1f}s)")
                    return verified_data, errors, url
                
                # For Semantic Scholar, try additional retries with increasing delays
                if api_name == 'semantic_scholar' and not success:
                    for retry_attempt in range(2):  # Additional 2 retries for Semantic Scholar
                        retry_delay = delay * (self.retry_backoff_factor ** (retry_attempt + 1))
                        retry_delay = min(retry_delay, self.max_retry_delay)
                        retry_jitter = retry_delay * 0.25 * (2 * random.random() - 1)
                        final_retry_delay = max(1.0, retry_delay + retry_jitter)
                        
                        print(f"      🔄 RETRY: S2 attempt {retry_attempt + 2} after {final_retry_delay:.1f}s")
                        logger.debug(f"Enhanced Hybrid: Additional Semantic Scholar retry {retry_attempt + 2} after {final_retry_delay:.1f}s")
                        time.sleep(final_retry_delay)
                        
                        verified_data, errors, url, success, _ = self._try_api(api_name, api_instance, reference, is_retry=True)
                        if success:
                            logger.debug(f"Enhanced Hybrid: {api_name} succeeded on retry {retry_attempt + 2} (delay: {final_retry_delay:.1f}s)")
                            return verified_data, errors, url
        
        # PHASE 3: If all APIs failed or returned incomplete data, use best available incomplete data as fallback
        incomplete_results = [r for r in [crossref_result, openalex_result] if r is not None]
        if incomplete_results:
            # Prefer CrossRef over OpenAlex for incomplete data (usually more reliable)
            best_incomplete = crossref_result if crossref_result else openalex_result
            logger.debug("Enhanced Hybrid: No complete data found, using incomplete data as fallback")
            return best_incomplete
        
        # If all APIs failed, return unverified
        failed_count = len(failed_apis)
        total_attempted = (1 if self.local_db else 0) + (1 if self.semantic_scholar else 0) + (1 if self.openalex else 0) + (1 if self.crossref else 0)
        
        if failed_count > 0:
            logger.debug(f"Enhanced Hybrid: All {total_attempted} APIs failed to verify reference ({failed_count} retried)")
        else:
            logger.debug("Enhanced Hybrid: All available APIs failed to verify reference")
            
        return None, [{'error_type': 'unverified', 'error_details': 'Could not verify reference using any available API'}], None
    
    def verify_references_batch(self, references: List[Dict[str, Any]]) -> Tuple[List[Tuple[Optional[Dict], bool, str]], Dict[str, int]]:
        """
        Verify multiple references efficiently using batch API calls.
        
        Strategy:
        1. Use OpenAlex batch search for all titles (most efficient)
        2. Use Semantic Scholar batch for papers not found in OpenAlex  
        3. Fall back to individual verification for remaining papers
        
        Args:
            references: List of reference dicts, each with 'title', 'year', 'authors', 'url'
            
        Returns:
            Tuple of:
            - List of (verified_paper, is_valid, reason) tuples, one per input
            - Stats dict with 'total', 'verified', 'not_found', 'throttled' counts
        """
        logger.info(f"Enhanced Hybrid: Starting batch verification for {len(references)} papers")
        
        stats = {
            'total': len(references),
            'verified': 0,
            'not_found': 0,
            'throttled': 0,
            'openalex_found': 0,
            'semantic_scholar_found': 0,
            'individual_fallback': 0
        }
        
        # Results list to match input order
        results = [None] * len(references)
        unfound_indices = list(range(len(references)))
        
        # PHASE 1: Try OpenAlex batch search (most efficient - can handle 20+ titles per call)
        if self.openalex and hasattr(self.openalex, 'search_works_batch'):
            logger.info("Enhanced Hybrid: Phase 1 - OpenAlex batch search")
            try:
                openalex_results, openalex_stats = self.openalex.search_works_batch(references)
                stats['throttled'] += openalex_stats.get('throttled', 0)
                
                # Process OpenAlex results
                for idx in list(unfound_indices):
                    work_data = openalex_results.get(idx)
                    if work_data:
                        # Extract URL and build result
                        url = self.openalex.extract_url_from_work(work_data) if hasattr(self.openalex, 'extract_url_from_work') else None
                        paper = references[idx].copy()
                        paper['url'] = url
                        paper['title'] = work_data.get('title', work_data.get('display_name', paper.get('title')))
                        paper['link_valid'] = True
                        paper['external_ids'] = work_data.get('ids', {})
                        
                        results[idx] = (paper, True, "Verified by OpenAlex batch")
                        stats['verified'] += 1
                        stats['openalex_found'] += 1
                        unfound_indices.remove(idx)
                
                logger.info(f"Enhanced Hybrid: OpenAlex batch found {stats['openalex_found']}/{len(references)} papers")
                
            except Exception as e:
                print(f"      ⚠️ OpenAlex batch search failed: {e}")
                logger.warning(f"Enhanced Hybrid: OpenAlex batch search failed: {e}")
        
        # PHASE 2: Try Semantic Scholar batch for remaining papers
        s2_batch_throttled = False  # Track if S2 batch was throttled to skip individual fallback
        
        if unfound_indices and self.semantic_scholar and hasattr(self.semantic_scholar, 'search_papers_batch'):
            logger.info(f"Enhanced Hybrid: Phase 2 - Semantic Scholar batch search for {len(unfound_indices)} remaining papers")
            
            # Build subset of references for unfound indices
            unfound_refs = [references[i] for i in unfound_indices]
            
            # Retry loop with exponential backoff for throttling
            max_batch_retries = 3
            batch_backoff = 10.0  # Start with 10 seconds (aggressive backoff)
            
            for batch_attempt in range(max_batch_retries):
                try:
                    ss_results, ss_stats = self.semantic_scholar.search_papers_batch(unfound_refs)
                    stats['throttled'] += ss_stats.get('throttled', 0)
                    
                    # Check if we got throttled during the batch (throttle count > 0 means retries happened)
                    if ss_stats.get('throttled', 0) > 0:
                        s2_batch_throttled = True
                    
                    # Process Semantic Scholar results
                    for i, idx in enumerate(unfound_indices[:]):  # Copy to allow modification
                        paper_data = ss_results.get(i)
                        if paper_data:
                            # Extract URL from paper data
                            external_ids = paper_data.get('externalIds', {})
                            url = None
                            
                            # Priority: PubMed > DOI > arXiv > Semantic Scholar URL
                            if external_ids.get('PubMed') or external_ids.get('PMID'):
                                pmid = external_ids.get('PubMed') or external_ids.get('PMID')
                                url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                            elif external_ids.get('DOI'):
                                url = f"https://doi.org/{external_ids['DOI']}"
                            elif paper_data.get('openAccessPdf', {}).get('url'):
                                url = paper_data['openAccessPdf']['url']
                            elif external_ids.get('ArXiv'):
                                url = f"https://arxiv.org/abs/{external_ids['ArXiv']}"
                            elif paper_data.get('url'):
                                url = paper_data['url']
                            
                            if url:
                                paper = references[idx].copy()
                                paper['url'] = url
                                paper['title'] = paper_data.get('title', paper.get('title'))
                                paper['link_valid'] = True
                                paper['external_ids'] = external_ids
                                
                                results[idx] = (paper, True, "Verified by Semantic Scholar batch")
                                stats['verified'] += 1
                                stats['semantic_scholar_found'] += 1
                                unfound_indices.remove(idx)
                    
                    logger.info(f"Enhanced Hybrid: Semantic Scholar batch found {stats['semantic_scholar_found']} more papers")
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    error_str = str(e).lower()
                    is_throttle = "429" in str(e) or "rate limit" in error_str or "too many requests" in error_str
                    
                    if is_throttle and batch_attempt < max_batch_retries - 1:
                        # Exponential backoff with jitter
                        wait_time = max(5.0, batch_backoff * (2 ** batch_attempt) + random.random())
                        print(f"      ⚠️ S2 batch throttled, waiting {wait_time:.1f}s before retry {batch_attempt + 2}/{max_batch_retries}")
                        logger.warning(f"Enhanced Hybrid: S2 batch throttled, retrying in {wait_time:.1f}s")
                        time.sleep(wait_time)
                        s2_batch_throttled = True
                    else:
                        print(f"      ⚠️ Semantic Scholar batch search failed: {e}")
                        logger.warning(f"Enhanced Hybrid: Semantic Scholar batch search failed: {e}")
                        if is_throttle:
                            s2_batch_throttled = True
                        break  # Non-throttle error or max retries reached
        
        # PHASE 3: Individual fallback for remaining papers (if any)
        # IMPORTANT: Skip individual fallback if S2 batch was throttled to avoid cascading failures
        if unfound_indices and not s2_batch_throttled:
            logger.info(f"Enhanced Hybrid: Phase 3 - Individual verification for {len(unfound_indices)} remaining papers")
            
            for idx in unfound_indices:
                ref = references[idx]
                try:
                    verified_data, errors, url = self.verify_reference(ref)
                    
                    if verified_data and url:
                        paper = ref.copy()
                        paper['url'] = url
                        paper['title'] = verified_data.get('title', paper.get('title'))
                        paper['link_valid'] = True
                        paper['external_ids'] = verified_data.get('externalIds', {})
                        
                        results[idx] = (paper, True, "Verified by individual API call")
                        stats['verified'] += 1
                        stats['individual_fallback'] += 1
                    else:
                        # Not found
                        error_msg = "No valid URL found" if errors else "Paper not found in any API"
                        results[idx] = (ref, False, f"Verification failed: {error_msg}")
                        stats['not_found'] += 1
                        
                except Exception as e:
                    results[idx] = (ref, False, f"Exception during verification: {e}")
                    stats['not_found'] += 1
        elif unfound_indices and s2_batch_throttled:
            # S2 was throttled, skip individual calls to avoid making it worse
            print(f"      ⏸️ Skipping individual fallback for {len(unfound_indices)} papers (S2 throttled)")
            logger.warning(f"Enhanced Hybrid: Skipping individual fallback due to S2 throttling, {len(unfound_indices)} papers marked not found")
            for idx in unfound_indices:
                results[idx] = (references[idx], False, "Verification skipped: S2 API throttled")
                stats['not_found'] += 1
        
        # Mark any remaining None results as not found
        for idx, result in enumerate(results):
            if result is None:
                results[idx] = (references[idx], False, "Verification failed: Paper not found")
                stats['not_found'] += 1
        
        logger.info(f"Enhanced Hybrid: Batch verification complete - "
                   f"{stats['verified']}/{stats['total']} verified "
                   f"(OpenAlex: {stats['openalex_found']}, S2: {stats['semantic_scholar_found']}, "
                   f"Individual: {stats['individual_fallback']}), "
                   f"Not found: {stats['not_found']}, Throttled: {stats['throttled']}")
        
        return results, stats
    
    def _try_openreview_search(self, reference: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]], Optional[str], bool, str]:
        """
        Try to verify reference using OpenReview search
        
        Returns:
            Tuple of (verified_data, errors, url, success, failure_type)
        """
        if not self.openreview:
            return None, [], None, False, 'none'
        
        start_time = time.time()
        failure_type = 'none'
        
        try:
            verified_data, errors, url = self.openreview.verify_reference_by_search(reference)
            duration = time.time() - start_time
            
            # Consider it successful if we found data or verification errors
            success = verified_data is not None or len(errors) > 0
            self._update_api_stats('openreview', success, duration)
            
            if success:
                logger.debug(f"Enhanced Hybrid: OpenReview search successful in {duration:.2f}s, URL: {url}")
                return verified_data, errors, url, True, 'none'
            else:
                logger.debug(f"Enhanced Hybrid: OpenReview search found no results in {duration:.2f}s")
                return None, [], None, False, 'not_found'
                
        except requests.exceptions.Timeout as e:
            duration = time.time() - start_time
            self._update_api_stats('openreview', False, duration)
            failure_type = 'timeout'
            logger.debug(f"Enhanced Hybrid: OpenReview search timed out in {duration:.2f}s: {e}")
            return None, [], None, False, failure_type
            
        except requests.exceptions.RequestException as e:
            duration = time.time() - start_time
            self._update_api_stats('openreview', False, duration)
            
            # Check if it's a rate limiting error
            if hasattr(e, 'response') and e.response is not None:
                if e.response.status_code in [429, 503]:
                    failure_type = 'throttled'
                elif e.response.status_code >= 500:
                    failure_type = 'server_error'
                else:
                    failure_type = 'other'
            else:
                failure_type = 'other'
            
            logger.debug(f"Enhanced Hybrid: OpenReview search failed in {duration:.2f}s: {type(e).__name__}: {e}")
            return None, [], None, False, failure_type
            
        except Exception as e:
            duration = time.time() - start_time
            self._update_api_stats('openreview', False, duration)
            failure_type = 'other'
            logger.debug(f"Enhanced Hybrid: OpenReview search error in {duration:.2f}s: {type(e).__name__}: {e}")
            return None, [], None, False, failure_type
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for all APIs
        
        Returns:
            Dictionary with performance statistics
        """
        stats = {}
        for api_name, api_stats in self.api_stats.items():
            total_calls = api_stats['success'] + api_stats['failure']
            if total_calls > 0:
                success_rate = api_stats['success'] / total_calls
                stats[api_name] = {
                    'success_rate': success_rate,
                    'total_calls': total_calls,
                    'avg_time': api_stats['avg_time'],
                    'success_count': api_stats['success'],
                    'failure_count': api_stats['failure']
                }
            else:
                stats[api_name] = {
                    'success_rate': 0,
                    'total_calls': 0,
                    'avg_time': 0,
                    'success_count': 0,
                    'failure_count': 0
                }
        return stats
    
    def log_performance_summary(self):
        """Log a summary of API performance statistics (only if debug mode is enabled)"""
        if not self.debug_mode:
            return
            
        stats = self.get_performance_stats()
        logger.info("Enhanced Hybrid API Performance Summary:")
        for api_name, api_stats in stats.items():
            if api_stats['total_calls'] > 0:
                logger.info(f"  {api_name}: {api_stats['success_rate']:.2%} success rate, "
                           f"{api_stats['total_calls']} calls, {api_stats['avg_time']:.2f}s avg")
            else:
                logger.info(f"  {api_name}: not used")
    
    def normalize_paper_title(self, title: str) -> str:
        """
        Normalize paper title for comparison (delegates to Semantic Scholar checker)
        """
        if self.semantic_scholar:
            return self.semantic_scholar.normalize_paper_title(title)
        else:
            # Basic normalization if Semantic Scholar is not available
            import re
            title = re.sub(r'\s+', ' ', title.strip().lower())
            return re.sub(r'[^\w\s]', '', title)
    
    def compare_authors(self, cited_authors: List[str], correct_authors: List[Any]) -> Tuple[bool, str]:
        """
        Compare author lists (delegates to shared utility)
        """
        from utils.text_utils import compare_authors
        return compare_authors(cited_authors, correct_authors)

# Backward compatibility alias
HybridReferenceChecker = EnhancedHybridReferenceChecker