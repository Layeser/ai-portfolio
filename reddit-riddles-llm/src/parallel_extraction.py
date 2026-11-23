"""
Parallel processing functions for extracting problem/solution pairs.
This module provides optimized parallel processing for LLM-based answer selection.
"""

import concurrent.futures
import re

import pandas as pd
from tqdm import tqdm


def process_single_post(post_data, comments_by_post, use_top_level_only, strategy, llm_model, select_best_answer, select_best_answer_with_llm):
    '''
    Processes a single post to extract problem/solution pair.
    
    Helper function for parallel processing.
    
    :param post_data: Tuple of (post_id, post_row) where post_row is a Series
    :param comments_by_post: GroupBy object with comments grouped by post_id
    :param use_top_level_only: If True, only consider top-level comments
    :param strategy: Strategy for selecting best answer
    :param llm_model: Model identifier for OpenRouter
    :param select_best_answer: Function for score-based selection
    :param select_best_answer_with_llm: Function for LLM-based selection
    :type post_data: Tuple[str, pd.Series]
    :type comments_by_post: pd.core.groupby.DataFrameGroupBy
    :type use_top_level_only: bool
    :type strategy: str
    :type llm_model: str
    :return: Dictionary with problem/solution pair or None, and rejection status
    :type: Tuple[Optional[Dict], bool]
    '''
    post_id, post_row = post_data
    
    if post_id not in comments_by_post.groups:
        return None, False
    
    comments_for_post = comments_by_post.get_group(post_id)
    
    # Select best answer based on strategy
    if strategy == "llm_based":
        best_answer = select_best_answer_with_llm(
            post_row['riddle_text'], 
            comments_for_post, 
            use_top_level_only=use_top_level_only,
            model=llm_model
        )
        rejected = (best_answer is None)
    else:  # score_based (default)
        best_answer = select_best_answer(comments_for_post, use_top_level_only=use_top_level_only)
        rejected = False
    
    if best_answer is not None:
        # Clean answer text (remove Reddit spoiler tags)
        answer_text = best_answer['body']
        answer_text = re.sub(r'&gt;!', '', answer_text)
        answer_text = re.sub(r'!&lt;', '', answer_text)
        answer_text = re.sub(r'&lt;', '', answer_text)
        answer_text = re.sub(r'&gt;', '', answer_text)
        answer_text = re.sub(r'\[/?spoiler\]', '', answer_text, flags=re.IGNORECASE)
        answer_text = re.sub(r'\(?\s*spoiler\s*\)?', '', answer_text, flags=re.IGNORECASE)
        answer_text = re.sub(r'\*\*?spoiler\*\*?', '', answer_text, flags=re.IGNORECASE)
        answer_text = re.sub(r'(?i)\b(answer|solution|riddle answer|correct answer|my guess|i think it\'s)\b[:\-]?', '', answer_text)
        answer_text = re.sub(r'(?i)\bthe answer is\b[:\-]?', '', answer_text)
        answer_text = re.sub(r'(?i)\bthat would be\b[:\-]?', '', answer_text)
        answer_text = re.sub(r'\*{1,2}', '', answer_text)
        answer_text = re.sub(r'[_~>`#]', '', answer_text)
        answer_text = re.sub(r'\s+', ' ', answer_text)
        answer_text = answer_text.strip()
        
        return {
            'post_id': post_id,
            'problem': post_row['riddle_text'],
            'solution': answer_text,
            'solution_score': best_answer['score'],
            'solution_is_top_level': best_answer['is_top_level'],
            'num_comments': len(comments_for_post),
            'title': post_row['title'],
            'strategy': strategy
        }, rejected
    
    return None, rejected


def extract_problem_solution_pairs_parallel(posts_df, comments_df, use_top_level_only=True, 
                                             strategy="score_based", llm_model="openai/gpt-4o-mini",
                                             sample_size=None, max_workers=10,
                                             filter_solved_posts_func=None,
                                             prepare_comments_func=None,
                                             select_best_answer_func=None,
                                             select_best_answer_with_llm_func=None):
    '''
    Extracts problem/solution pairs from posts and comments with parallel processing support.
    
    This is the optimized version that uses ThreadPoolExecutor for parallel API calls.
    Replace the original extract_problem_solution_pairs function with this implementation.
    
    :param posts_df: DataFrame with filtered solved posts
    :param comments_df: DataFrame with processed comments
    :param use_top_level_only: If True, only consider top-level comments (default = True)
    :param strategy: Strategy for selecting best answer - "score_based" or "llm_based" (default = "score_based")
    :param llm_model: Model identifier for OpenRouter when using "llm_based" strategy (default = "openai/gpt-4o-mini")
    :param sample_size: Number of posts to sample (None = process all posts)
    :param max_workers: Maximum number of parallel workers for LLM strategy (default = 10)
    :param filter_solved_posts_func: Function to filter solved posts
    :param prepare_comments_func: Function to prepare comments
    :param select_best_answer_func: Function for score-based answer selection
    :param select_best_answer_with_llm_func: Function for LLM-based answer selection
    :return: DataFrame with problem/solution pairs
    '''
    
    # Filter solved posts
    solved_posts = filter_solved_posts_func(posts_df)
    print(f"Filtered to {len(solved_posts)} solved self-posts")
    
    # Apply sampling if requested
    if sample_size is not None and sample_size < len(solved_posts):
        solved_posts = solved_posts.head(sample_size).copy()
        print(f"Processing {len(solved_posts)} posts")
    
    # Prepare comments
    valid_comments = prepare_comments_func(comments_df)
    print(f"Valid comments: {len(valid_comments)}")
    
    # Group comments by post_id
    comments_by_post = valid_comments.groupby('post_id')
    
    # Extract problem/solution pairs
    problem_solution_pairs = []
    rejected_posts = 0
    
    print(f"Using strategy: {strategy}")
    
    # Prepare post data as list of tuples
    post_data_list = [(row['post_id'], row) for _, row in solved_posts.iterrows()]
    
    if strategy == "llm_based":
        # Use parallel processing for LLM strategy
        print(f"Using parallel processing with {max_workers} workers...")
        print("Note: Adjust max_workers based on your API rate limits")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_post = {
                executor.submit(
                    process_single_post, 
                    post_data, 
                    comments_by_post, 
                    use_top_level_only, 
                    strategy, 
                    llm_model,
                    select_best_answer_func,
                    select_best_answer_with_llm_func
                ): post_data[0] 
                for post_data in post_data_list
            }
            
            # Process completed tasks with progress bar
            with tqdm(total=len(post_data_list), desc="Extracting pairs (parallel)") as pbar:
                for future in concurrent.futures.as_completed(future_to_post):
                    post_id = future_to_post[future]
                    try:
                        result, rejected = future.result()
                        if result is not None:
                            problem_solution_pairs.append(result)
                        if rejected:
                            rejected_posts += 1
                    except Exception as e:
                        print(f"\nError processing post {post_id}: {e}")
                    pbar.update(1)
    else:
        # Sequential processing for score-based strategy (fast enough)
        for post_data in tqdm(post_data_list, desc="Extracting pairs"):
            result, rejected = process_single_post(
                post_data, 
                comments_by_post, 
                use_top_level_only, 
                strategy, 
                llm_model,
                select_best_answer_func,
                select_best_answer_with_llm_func
            )
            if result is not None:
                problem_solution_pairs.append(result)
            if rejected:
                rejected_posts += 1
    
    pairs_df = pd.DataFrame(problem_solution_pairs)
    print(f"\nExtracted {len(pairs_df)} problem/solution pairs")
    if strategy == "llm_based" and rejected_posts > 0:
        print(f"LLM rejected {rejected_posts} posts (no satisfactory answer found)")
        if len(post_data_list) > 0:
            print(f"Rejection rate: {rejected_posts / len(post_data_list) * 100:.1f}%")
    
    return pairs_df

