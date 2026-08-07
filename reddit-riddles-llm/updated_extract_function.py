"""
Complete updated extract_problem_solution_pairs function with parallel processing support.
Copy this function to replace the existing one in your notebook.
"""

def extract_problem_solution_pairs(posts_df, comments_df, use_top_level_only=True, 
                                     strategy="score_based", llm_model="openai/gpt-4o-mini",
                                     sample_size=None, max_workers=10):
    '''
    Extracts problem/solution pairs from posts and comments.
    
    Supports multiple strategies for selecting the best answer:
    - "score_based": Uses highest scoring comment (default)
    - "llm_based": Uses LLM to evaluate and select best answer, can reject if none are good
    
    When using "llm_based" strategy, processes posts in parallel using ThreadPoolExecutor
    for much faster execution (5-15x speedup depending on max_workers).
    
    :param posts_df: DataFrame with filtered solved posts
    :param comments_df: DataFrame with processed comments
    :param use_top_level_only: If True, only consider top-level comments (default = True)
    :param strategy: Strategy for selecting best answer - "score_based" or "llm_based" (default = "score_based")
    :param llm_model: Model identifier for OpenRouter when using "llm_based" strategy (default = "openai/gpt-4o-mini")
    :param sample_size: Number of posts to sample (None = process all posts)
    :param max_workers: Maximum number of parallel workers for LLM strategy (default = 10)
                        Adjust based on API rate limits. Higher = faster but may hit rate limits.
    :type posts_df: pd.DataFrame
    :type comments_df: pd.DataFrame
    :type use_top_level_only: bool
    :type strategy: str
    :type llm_model: str
    :type sample_size: Optional[int]
    :type max_workers: int
    :return: DataFrame with problem/solution pairs
    :type: pd.DataFrame
    '''
    # Filter solved posts
    solved_posts = filter_solved_posts(posts_df)
    print(f"Filtered to {len(solved_posts)} solved self-posts")
    
    # Apply sampling if requested
    if sample_size is not None and sample_size < len(solved_posts):
        solved_posts = solved_posts.head(sample_size).copy()
        print(f"Processing {len(solved_posts)} posts")
    
    # Prepare comments
    valid_comments = prepare_comments(comments_df)
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
                    llm_model
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
                llm_model
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

