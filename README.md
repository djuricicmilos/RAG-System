# Building Our Own RAG Model

Artificial Intelligence is evolving at lightning speed, and one of the hottest concepts powering real-world applications today is **RAG — Retrieval-Augmented Generation**.

Sounds technical? Don't worry. In this blog, we'll break it down in **simple terms**, walk you through **how we built our own RAG model**, and show you why it's such a **game-changer** in the AI space.

## What is RAG (Retrieval-Augmented Generation)?

At its core, RAG is about making AI systems **smarter and more reliable**.

Without RAG, a Large Language Model (LLM) answers based only on what it learned during training. That often leads to **hallucinations** — answers that *sound* correct but aren't grounded in real data.

With RAG:

* **Retrieval**: The system pulls relevant chunks of information from a custom **data store** (like your documents, PDFs, or databases).
* **Augmented Generation**: These retrieved chunks are added to the **prompt** and passed to the LLM for **text generation**.

This combination gives **factual, context-rich, and optimized responses**.

Think of RAG as an **AI pipeline** that reduces the hallucination of LLMs by connecting them directly to your **source of truth**.

## Our Project

To make things practical, we decided to **build a RAG model based on our resume and personal data**.

* Retrieves relevant chunks from our docs
* Parses the data to answer queries about our docs
* Generates clear, context-aware answers

Now, instead of scrolling through our resume, you can simply ask:

* "What is a rule of thumb when using A2A?"
* "Give me key concepts of the RAG design pattern."

And the bot will respond instantly with accurate, grounded answers.

## Architecture

Let's break down the **end-to-end RAG pipeline**.

### 1️. Data Ingestion & Parsing

The process starts with **data ingestion**. We fed our **resume PDF** and personal documents into the system.

Using **LangChain's loader**, we performed **data parsing** to extract meaningful text.

### 2️. Chunking the Data

Large documents aren't efficient to search directly. So the parsed data was split into **chunks** (small pieces of text).

Why chunking matters:

* Each chunk is easier to retrieve later
* It keeps responses accurate and focused
* Prevents the model from being overloaded with irrelevant context

### 3️. Embeddings: Turning Text into Vectors

We used the **all-MiniLM-L6-v2 model** to convert chunks into **embeddings** — numerical representations of text.

This step makes the text **searchable in vector space**. So when a user query comes in, the system can quickly find **similar embeddings** (aka the most relevant chunks).

### 4️. Vector Store & Retriever

We stored these embeddings in **ChromaDB** — an open-source **vector store** optimized for similarity search.

* **Retriever**: Acts as the "search engine" of the pipeline.
* It finds the top chunks that best match the user query.

Alternative: FAISS (another popular open-source vector store).

### 5️. Query + Context + Prompt → LLM

Here's where the magic happens:

1. The **user query** is embedded
2. The **retriever** pulls the top-matching chunks from the vector store
3. These chunks are added as **context** to the **prompt**
4. The **LLM API** generates a response based on both query + retrieved data

This step ensures the **text generation** is grounded in reality, reducing hallucination and optimizing the quality of answers.

## Optimization & Open-Source Options

Everything in this setup is **open-source**.

* For embeddings: **all-MiniLM-L6-v2** (open-source, lightweight)
* For vector store: **ChromaDB** (fast, reliable)
* For pipeline orchestration: **LangChain** (flexible, modular)
* For text generation: We used **Ollama** with LLaMA models.

This makes the solution **cost-effective** and highly customizable.

## Final Thoughts

RAG is the **bridge between static knowledge in LLMs and dynamic, real-world information**. By combining **retrieval, chunking, embeddings, vector stores, and optimized text generation**, we can build AI systems that are not only smart but also **trustworthy**.

The future of AI isn't just about models — it's about **data ingestion, retrieval, and grounding**. And **RAG** is leading the way.
