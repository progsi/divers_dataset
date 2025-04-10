import os
from youtubesearchpython import VideosSearch

OUTPUT_DIR = os.path.join("data", "youtube")

def search(q: str, max_results: int = 500, max_pages: int = 100) -> list:
    """
    Search for videos on YouTube.
    Args:
        q (str): The search query.
        max_results (int): The maximum number of results to return.
    Returns:
        list: A list of dictionaries containing video information.
    """
    results = []
    search = VideosSearch(q, limit = max_results)

    cur_pages = 0
    results_left = True
    while results_left and len(results) < max_results and cur_pages < max_pages:
        # collect results
        new_results = search.result()["result"]
        print(f"{len(new_results)} found")
        results += new_results
        if len(new_results) == 0:
            print("No more results")
            break
        
        # iterate to next page
        results_left = search.next()
        cur_pages += 1

    results = results[:max_results]
    return results
