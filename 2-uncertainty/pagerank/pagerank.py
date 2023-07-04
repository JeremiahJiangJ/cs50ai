import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    has_outgoing_links = (len(corpus[page]) >= 1)

    if has_outgoing_links:
        prob_dist = calculate_probability_distribution(corpus, page, damping_factor)
    else:
        prob_dist = calculate_probability_distribution(corpus)

    return prob_dist

def calculate_probability_distribution(corpus, page="", damping_factor=0.):
    '''
    If given a page:
        Returns the probability distribution over which page to visit next 
    else:
        Returns a probability distribution that chooses randomly among all 
        pages with equal probability
    '''
    prob_dist = {}
    len_corpus = len(corpus.keys())
    
    calculate_at_random = (len(page) == 0)

    if calculate_at_random:
        for page_i in corpus.keys():
            prob_dist[page_i] = 1/len_corpus

    else:
        len_page = len(corpus[page])
        random_factor = (1 - damping_factor) / len_corpus

        for page_i in corpus.keys():
            if page_i not in corpus[page]:
                prob_dist[page_i] = random_factor

            else:
                prob_dist[page_i] = random_factor + (damping_factor/len_page)

    return prob_dist
        


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    page_rank = dict.fromkeys(corpus.keys(), 0)

    sample = None

    for _ in range(n):
        if sample is None:
            sample = random.choice(list(corpus.keys()))
        else:
            prob_dist = transition_model(corpus, sample, damping_factor)
            page_names = list(prob_dist.keys())
            page_weights = [prob_dist[i] for i in prob_dist]
            sample = random.choices(page_names, page_weights, k = 1)[0]

        page_rank[sample] += 1

    for item in page_rank:
        page_rank[item] /= n

    # Debug message if somehow the probabilities differ greatly from 1
    if abs(1 - sum(page_rank.values())) >= 0.00001:
        print("WARNING: PageRank Values do not sum up to 1\n")

    return page_rank


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    len_corpus = len(corpus)
    prev_page_rank = {}
    curr_page_rank = {}

    # Initialize each page with a rank of 1/length of corpus
    for page_name in corpus:
        prev_page_rank[page_name] = 1/len_corpus

    res = {}

    while True:
        for page in corpus:
            link_weight = 0

            for page_link in corpus:

                if len(corpus[page_link]) == 0:
                    link_weight += (prev_page_rank[page_link] / len(corpus))

                if page in corpus[page_link]:
                    link_weight += (prev_page_rank[page_link] / len(corpus[page_link]))

            link_weight *= damping_factor
            link_weight += (1 - damping_factor) / len_corpus

            curr_page_rank[page] = link_weight

        difference = max([abs(curr_page_rank[i] - prev_page_rank[i]) for i in prev_page_rank])

        if difference < 0.001:
            res = prev_page_rank.copy()
            break

        prev_page_rank = curr_page_rank.copy()

    # Debug message if somehow the probabilities differ greatly from 1
    if abs(1 - sum(res.values())) >= 0.00001:
        print("WARNING: PageRank Values do not sum up to 1\n")

    return res
if __name__ == "__main__":
    main()
