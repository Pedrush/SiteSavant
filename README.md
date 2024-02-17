# SiteSavant

### *A chatbot for any website on-demand.*

SiteSavant is a chatbot application designed to provide on-demand information from any website. It works by scraping the content of a target website and initializing a chatbot over this data. The app leverages vector similarity techniques to fetch the most relevant data in response to user queries. This ensures that each interaction with the chatbot is informed by the context information from the website, enabling accurate and relevant answers. In essence, SiteSavant is a [Retrieval-Augmented Generation (RAG)](https://arxiv.org/abs/2005.11401) app with the added complexity of web scraping.


https://github.com/Glueish/SiteSavant/assets/87642985/001e5a44-70a1-4139-a5cf-4a76869cafca


## Detailed workflow

### 1. **Web Scraping**
The app automatically collects data from a specified URL, continuing to scrape the website for text information up to a user-defined depth. Functionalities are defined and documented in the `services/website_scraper.py`

### 2. **Creating Embeddings**
Next, the collected text segments are transformed into high-dimensional vectors through a process called "embedding". This step converts the text into a numerical format that represents its meaning, facilitating efficient retrieval of information later. This feature currently integrates exclusively with the [Cohere](https://cohere.com/) API, with its functions detailed in `services/embeddings_creator.py`.

### 3. **Removing Duplicate Information**
Given the diversity of web content and layout, scraping often captures unnecessary elements like menus, ads, and footers, leading to redundant information. This stage cleans up such duplicate information, improving data quality, especially when identical text segments are repeatedly scraped from the site. Duplication elimination relies on the [Facebook AI Similarity Search (FAISS)](https://ai.meta.com/tools/faiss/) library, outlined in `services/embeddings_deduplicator.py`.

### 4. **Indexing Embeddings**
After cleaning and embedding the text, these vectors are put into a database (called "index") for easy access. This step is crucial for efficiently storing and retrieving data. This feature currently integrates exclusively with the [Pinecone](https://www.pinecone.io/) API, with its functions detailed in `services/embeddings_indexer.py`.

### 5. **Chatbot Interaction**
Now, users can enter their questions. These are first converted into a numerical format (embedded) using the [Cohere](https://cohere.com/) API. This numerical data is then matched (using vector similarity) against the information stored in the [Pinecone](https://www.pinecone.io/) index to find the most relevant content. This content, along with the user's original question, is provided to the chosen Large Language Model (LLM) for processing. The LLM uses this data to generate responses that aim to accurately address the user's query. Currently the project supports only [OpenAI](https://openai.com/) API integration. Functionalities are defined and documented in the `services/query_handler.py` and `services/chatbot_interactor.py`


## Getting Started

This guide will help you set up and run SiteSavant on your local machine for development and testing purposes. Follow these steps to get started.

### Prerequisites

- Python 3.10.
- [Poetry](https://python-poetry.org/) for Python package management.
- API keys from [Cohere](https://cohere.com/), [Pinecone](https://www.pinecone.io/), and [OpenAI](https://openai.com/).

### Installation

**Clone the Repository**

First, clone the SiteSavant repository to your local machine:

```bash
git clone https://github.com/glueish/sitesavant.git
cd sitesavant
```

**Install Dependencies with Poetry**

When cloning an existing repository that uses Poetry, the `pyproject.toml` and `poetry.lock` files define the project's dependencies. Therefore, running `poetry init` is not required. Simply proceed with installing the dependencies as outlined below.

Install the project dependencies using Poetry:

```bash
poetry install
```

This command will create a virtual environment and install all the necessary packages.

**Activate the Virtual Environment**
```bash
poetry shell
```

This command activates the project's virtual environment, allowing you to run scripts and commands within the isolated environment created and managed by Poetry.

**Set Up Environment Variables**

Create a `.env` file in the project's root directory of the project to store your API keys securely:

```plaintext
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
COHERE_API_KEY=your_cohere_api_key
```

### SiteSavant Configuration

Detailed customization of SiteSavant behaviour is possible via `config/parameters.yml`. 
```yaml
website_scraper:
  request_delay: 1
  max_depth: 2
  ...
```

Please remember to specify the `user_agent` paramter which essentially is how you introduce yourself to the website (e.g. "SiteSavant/0.1 (RAG-LLM Hobby Project; Contact: john.doe@example.com)")

### Additional Notes

- **Adding to PATH:** If you're using Windows Subsystem for Linux (WSL), ensure that the Python and Poetry executable paths are added to your system's PATH variable. This enables you to run Python and Poetry commands from the terminal. Neither Python nor Poetry will work otherwise.
- **Windows Compatibility:** SiteSavant is developed on WSL and should be compatible with Windows systems. However, if you encounter issues, consider running it within WSL or a Linux VM.

Feel free to contact me via sitesavant@outlook.com when facing problems with setting up the project :)

## Usage

Interact with SiteSavant using the command line. The application supports two main commands:

**Scrape:** This command fetches and prepares data from the specified website using the starting URL. It then initializes a chatbot that utilizes this data.

```bash
sitesavant scrape https://example.com
```

**Chat:** This command initializes the chatbot immediately. It starts the chatbot using the data that was last scraped and prepared.

```bash
sitesavant chat
```


## License
This project is licensed under the MIT License - see the `LICENSE.txt` file for details.


## Ethical Guidelines for Web Scraping
The `robots.txt` file is a text file webmasters create to instruct web crawlers. For example, it might specify which subdirectories within the website are not permitted to be fetched. These guidelines are placed in the root directory of the website (e.g. https://wikipedia.org/robots.txt)

SiteSavant respects `robots.txt` of the visited websites. However, the user is free to choose the duration of the delay between consequtive requests. Be cautious not to set it up too low to avoid overloading the server hosting the website. Setting the request-delay to less than 1 second is discouraged.


## Further development directions
The quality of the Information Retrieval (IR) pipeline is the biggest bottleneck for providing quality answers. Therefore, most of the following ideas aim to improve this specific part of the application.

1. **Asynchronous Processing:** The limited speed of web scraping is a primary bottleneck in gathering information for the chatbot. However, significant improvements in data processing speed can be achieved by adopting asynchronous programming. This approach allows for processing the sites as soon as they are fetched, instead of waiting for the entire scraping process to complete.
1. **Hybrid Search:** Introduce hybrid search. Currently, the search is solely semantic. For an understanding of keyword versus semantic search, refer to [keyword vs semantic search](https://enterprise-knowledge.com/expert-analysis-keyword-search-vs-semantic-search-part-one/).
1. **Query Classification:** After the user's query is provided, process it to determine the type of search required (e.g., semantic search, search filtered by metadata, keyword search).
1. **Query Rewriting:** After the user's query is provided, optimize it by using a Large Language Model to [rewrite it](https://twitter.com/LangChainAI/status/1715769143042134026).
1. **Enhanced Data Deduplication:** Introduce keyword-based duplicate information removal. Currently, the removal is based solely on the semantic meanings captured by the embeddings.
1. **Multi-Stage IR Pipeline:** Information Retrieval (IR) is essentially a re-ranking process aimed at retrieving the most relevant text chunks without compromising on precision. To balance computational cost and the quality of the results, a well-known approach of a multi-stage IR pipeline could be utilized. This strategy begins with a computationally efficient first stage, designed to quickly go through data and narrow the search down to the top 100 text chunks. Following this, a second stage employs a more computationally intensive model, such as a [Cross-Encoder](https://www.sbert.net/examples/applications/cross-encoder/README.html), to precisely match queries with the most relevant text, culminating in a top few results. Cross-Encoders, despite their higher computational complexity compared to Bi-Encoders, offer greater accuracy, making them well-suited for this final refinement step. However, their intensive computational demands render them infeasible for processing large text corpora. The current solution utilizes the Bi-Encoder approach. Also, consider tools like contextual compression or Max Marginal Relevance from Langchain. Consider utilizing models that aim to balance computational complexity and search simplicity, like [Colbertv2](https://youtu.be/VrL7AbrY438?si=TQnLmDJ3nR2yZXz9&t=2063). 
1. **Introduce Redundancy:** Compressing the meaning of a paragraph into a vector is a lossy compression. It's fundamentally flawed because long paragraphs can dilute the meaning excessively, and too short paragraphs may not be meaningful enough themselves. Additionally, the embedded paragraph might end before completing important information. Solutions include embedding smaller chunks and attaching more surrounding context upon retrieval. Also, consider introducing redundancy in two ways: creating two indexes with embeddings of different lengths and searching both during query time. Moreover, consider a chunking strategy that allows chunks to overlap, adding another layer of redundancy.
1. **RAPTOR:** To enhance the contextual understanding of retrieved data, implement RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval [[Twitter summary](https://twitter.com/bindureddy/status/1753994930366930953)] [[Paper](https://arxiv.org/abs/2401.18059)].
1. **Lost in the Middle effect:** LLMs tend to pay less attention to information placed in the middle of the prompt (called [Lost in the Middle effect](https://cs.stanford.edu/~nfliu/papers/lost-in-the-middle.arxiv2023.pdf)). Thus, RAG applications could improve by positioning the most relevant, retrieved context information at the beginning and the end of the prompt. This strategy is essentially an additional reranking step. For instance, in terms of relevance, the ranked text chunks could be arranged in the prompt as 1, 4, 5, 3, 2 instead of the sequential 1, 2, 3, 4, 5.
1. **Evaluation:** Evaluate the Information Retrieval (IR) quality using out-of-the-box solutions such as Elasticsearch and managed vector stores like Pinecone. Utilize metrics such as Normalized Discounted Cumulative Gain (NDCG), Discounted Cumulative Gain (DCG), Mean Reciprocal Rank (MRR), Precision, Recall, and Average Precision/Recall for assessment. Consider Langsmith for experiment tracking.
1. **Scalable Framework:** Transition to a suitable framework enabling pipeline creation such as Luigi, Apache Airflow, or Kedro to enhance the project's scalability.
1. **Integrations:** Integrate with other vector database and model providers, potentially by enhancing integration with LangChain.
1. **LangChain:** Integrate closely with LangChain, leveraging its growing community and expanding features.
1. **Embed Summaries:** In some cases, embeddings can be improved by generating them from summarized text instead of raw text. For example,  information extracted from emails.
1. **Robust Testing:** Develop tests, in spite of their [nondeterministic and flaky nature](https://austinhenley.com/blog/copilotpainpoints.html#:~:text=Testing%20is%20fundamental%20to%20software%20development%20but%20arduous%20when%20LLMs%20are%20involved).

## Final remarks
The project was created for educational purposes. The idea behind it was to implement RAG pipeline from scratch. Hence, for example, there is no use of ready-made Langchain retrievers, text_splitters, pre-built integrations with vector stores, embedding services etc. In consequence, this enabled me to learn about the challenges of buidling such tools relatively from scratch and to explore different software design choices.
