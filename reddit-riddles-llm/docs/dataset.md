# Reddit Riddles Dataset - Documentation

## Overview

This document provides a comprehensive guide to understanding the Reddit Riddles dataset structure and the relationship between posts and comments, particularly focusing on how riddle answers are validated and marked as resolved.

**Dataset Source:** [sileod/reddit-riddles on Hugging Face](https://huggingface.co/datasets/sileod/reddit-riddles/viewer/submissions/train)

---

## Dataset Structure

The dataset consists of two main tables:
- **POSTS**: Contains riddle statements
- **COMMENTS**: Contains proposed answers to riddles

The relationship between posts and comments is established via `link_id` (in comments) which corresponds to the post ID.

---

## POSTS Table

| Field | Type | Description |
|-------|------|-------------|
| `author` | string | Reddit username of the person who posted the riddle |
| `created_utc` | float (timestamp) | Publication date and time of the post |
| `is_self` | boolean | Indicates if the post is a "self-post" (pure text) or external link. External links are ignored to avoid unexpected content |
| `link_flair_text` | string | Flair applied to the post (e.g., "Solved", "null", "Unsolved", "R1", etc.). Used to identify if the riddle is resolved |
| `num_comments` | int | Number of comments under the post |
| `permalink` | string | Relative URL of the post on Reddit (e.g., `/r/riddles/comments/...`) |
| `score` | int | Reddit score (upvotes - downvotes) |
| `selftext` | string | **Post content** → Contains the riddle statement |
| `title` | string | Post title (e.g., "What am I?"). Combining this with `selftext` can help LLMs better understand the riddle |
| `url` | string | Direct link to the post or related content. For self-posts, this is the Reddit post link itself |
| `upvote_ratio` | float | Upvote ratio. Provides insight into popularity and quality |

---

## COMMENTS Table

| Field | Type | Description |
|-------|------|-------------|
| `subreddit` | string | Subreddit name (typically "riddles") |
| `link_id` | string | ID of the parent post (same value as post_id in posts table). **Primary key for linking tables** |
| `id` | string | Unique identifier of the comment |
| `body` | string | **Comment text** → Contains the proposed answer |
| `score` | int | Comment score (upvotes - downvotes). The highest score often indicates the correct answer |
| `author` | string | Reddit username of the comment author |
| `created_utc` | float (timestamp) | Creation date and time of the comment |
| `parent_id` | string | ID of the message this comment replies to. If replying to the original post, this equals `link_id` |
| `permalink` | string | Relative URL of the comment on Reddit |

---

## Key Concepts

### Data Workflow

1. **Riddle Posting**: A user posts a riddle as a "self-post" in `/r/riddles`
   - The riddle statement is stored in `selftext`
   - The title provides additional context
   - The post receives a flair status (initially "Unsolved" or similar)

2. **Answer Proposals**: Community members comment with proposed answers
   - Answers are stored in the `body` field of comments
   - Comments are scored by upvotes/downvotes
   - Higher scores typically indicate more credible answers

3. **Resolution**: When the post author finds the correct answer
   - They mark the comment as "solved"
   - The post's `link_flair_text` changes to "Solved"
   - The correct answer is validated through community engagement

### Focus of Analysis

**Current analysis focuses exclusively on posts marked as "Solved"** to ensure data quality and reliability of answer-riddle pairs.

---

## Data Quality Notes

- **External links are excluded** from the analysis to avoid unexpected or irrelevant content
- **Only self-posts** (`is_self = true`) are considered
- **Resolved riddles only** are used for downstream analysis (`link_flair_text = "Solved"`)
- High comment scores often correlate with correct answers, but should be validated
---