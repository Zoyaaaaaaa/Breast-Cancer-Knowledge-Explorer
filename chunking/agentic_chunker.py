import time
import random
import uuid
from typing import Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from google.api_core import exceptions

class AgenticChunker:
    def __init__(self):
        self.chunks = {}
        self.id_truncate_limit = 5
        self.generate_new_metadata_ind = True
        self.print_logging = True
        self.max_retries = 5  # Maximum number of retries for API calls
        self.base_delay = 1.5  # Base delay in seconds
        self.max_delay = 60  # Maximum delay in seconds

        # Initialize the Gemini model API
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            api_key="AIzaSyAOkCBY-kR4giHrV6l8XrZzVha50I9f6ZM",
            temperature=0.4
        )
        

    def _exponential_backoff(self, retry_count: int) -> float:
        """Calculate delay for exponential backoff with jitter"""
        delay = min(self.base_delay * (2 ** retry_count) + random.random(), self.max_delay)
        if self.print_logging and retry_count > 0:
            print(f"Rate limited, retrying in {delay:.2f} seconds...")
        time.sleep(delay)
        return delay

    def _call_api_with_retry(self, runnable, input_data):
        """Helper method to handle API calls with retry logic"""
        last_error = None
        for retry in range(self.max_retries + 1):
            try:
                return runnable.invoke(input_data).content
            except exceptions.ResourceExhausted as e:
                last_error = e
                if retry < self.max_retries:
                    self._exponential_backoff(retry)
                else:
                    raise
            except Exception as e:
                last_error = e
                break
        
        raise Exception(f"API call failed after {self.max_retries} retries") from last_error

    def add_propositions(self, propositions):
        for proposition in propositions:
            self.add_proposition(proposition)
    
    def pretty_print_chunk_outline(self):
        print("Chunk Outline\n")
        print(self.get_chunk_outline())

    def get_chunk_outline(self):
        """Get a string which represents the chunks you currently have."""
        chunk_outline = ""
        for chunk_id, chunk in self.chunks.items():
            single_chunk_string = f"""Chunk ({chunk['chunk_id']}): {chunk['title']}\nSummary: {chunk['summary']}\n\n"""
            chunk_outline += single_chunk_string
        return chunk_outline
    
    def _find_relevant_chunk(self, proposition) -> Optional[str]:
        current_chunk_outline = self.get_chunk_outline()

        PROMPT = ChatPromptTemplate.from_messages([
            (
                "system",
                """Determine whether the following proposition should belong to one of the existing chunks.
                A proposition should belong to a chunk if their meaning, topic, or purpose is similar.
                If a chunk is a match, return the chunk ID. If not, return "No chunks".
                Output only the chunk ID or "No chunks". Do not include any explanation or punctuation.""",
            ),
            ("user", "Current Chunks:\n{current_chunk_outline}"),
            ("user", "Proposition:\n{proposition}"),
        ])

        runnable = PROMPT | self.llm
        response = self._call_api_with_retry(runnable, {
            "proposition": proposition,
            "current_chunk_outline": current_chunk_outline
        }).strip()

        # Return the chunk ID only if it matches expected ID length
        if len(response) == self.id_truncate_limit:
            return response
        return None

    def add_proposition(self, proposition):
        if self.print_logging:
            print(f"\nAdding: '{proposition}'")

        if len(self.chunks) == 0:
            if self.print_logging:
                print("No chunks, creating a new one")
            self._create_new_chunk(proposition)
            return

        chunk_id = self._find_relevant_chunk(proposition)
        
        if chunk_id:
            if self.print_logging:
                print(f"Chunk Found ({self.chunks[chunk_id]['chunk_id']}), adding to: {self.chunks[chunk_id]['title']}")
            self.add_proposition_to_chunk(chunk_id, proposition)
        else:
            if self.print_logging:
                print("No chunks found")
            self._create_new_chunk(proposition)

    def add_proposition_to_chunk(self, chunk_id, proposition):
        self.chunks[chunk_id]['propositions'].append(proposition)

        if self.generate_new_metadata_ind:
            self.chunks[chunk_id]['summary'] = self._update_chunk_summary(self.chunks[chunk_id])
            self.chunks[chunk_id]['title'] = self._update_chunk_title(self.chunks[chunk_id])

    def _update_chunk_summary(self, chunk):
        """Update chunk summary if a new proposition is added"""
        PROMPT = ChatPromptTemplate.from_messages([
            ("system", """
                You are the steward of a group of chunks which represent groups of sentences that talk about a similar topic.
                A new proposition was just added to one of your chunks. Generate a very brief 1-sentence summary for the chunk.
                Your summary should generalize the content, anticipating broader topics.
                Example:
                Input: Proposition: Greg likes to eat pizza.
                Output: This chunk contains information about the types of food Greg likes to eat.
            """),
            ("user", "Chunk's propositions:\n{proposition}\n\nCurrent chunk summary:\n{current_summary}"),
        ])

        runnable = PROMPT | self.llm
        return self._call_api_with_retry(runnable, {
            "proposition": "\n".join(chunk['propositions']),
            "current_summary": chunk['summary']
        })
    
    def _update_chunk_title(self, chunk):
        """Update chunk title if a new proposition is added"""
        PROMPT = ChatPromptTemplate.from_messages([
            ("system", """
                You are the steward of a group of chunks which represent groups of sentences that talk about a similar topic.
                A new proposition was just added to one of your chunks. Generate a very brief updated chunk title.
                Your title should summarize the chunk in a few words.
                Example:
                Input: Summary: This chunk is about dates and times that the author talks about.
                Output: Date & Times
            """),
            ("user", "Chunk's propositions:\n{proposition}\n\nChunk summary:\n{current_summary}\n\nCurrent chunk title:\n{current_title}"),
        ])

        runnable = PROMPT | self.llm
        return self._call_api_with_retry(runnable, {
            "proposition": "\n".join(chunk['propositions']),
            "current_summary": chunk['summary'],
            "current_title": chunk['title']
        })

    def _create_new_chunk(self, proposition):
        new_chunk_id = str(uuid.uuid4())[:self.id_truncate_limit]
        new_chunk_summary = self._get_new_chunk_summary(proposition)
        new_chunk_title = self._get_new_chunk_title(new_chunk_summary)

        self.chunks[new_chunk_id] = {
            'chunk_id': new_chunk_id,
            'propositions': [proposition],
            'title': new_chunk_title,
            'summary': new_chunk_summary,
            'chunk_index': len(self.chunks)
        }
        if self.print_logging:
            print(f"Created new chunk ({new_chunk_id}): {new_chunk_title}")

    def _get_new_chunk_summary(self, proposition):
        PROMPT = ChatPromptTemplate.from_messages([
            ("system", """
                Generate a brief 1-sentence summary for a new chunk based on the provided proposition.
                The summary should generalize the content and anticipate broader topics.
            """),
            ("user", "Determine the summary of the new chunk that this proposition will go into:\n{proposition}"),
        ])

        runnable = PROMPT | self.llm
        return self._call_api_with_retry(runnable, {"proposition": proposition})
    
    def _get_new_chunk_title(self, summary):
        PROMPT = ChatPromptTemplate.from_messages([
            ("system", """
                Generate a brief chunk title based on the provided chunk summary.
                The title should be succinct and representative of the chunk's content.
            """),
            ("user", "Determine the title of the chunk that this summary belongs to:\n{summary}"),
        ])

        runnable = PROMPT | self.llm
        return self._call_api_with_retry(runnable, {"summary": summary})

    def pretty_print_chunks(self):
        """Print all chunks with their propositions"""
        for chunk_id, chunk in self.chunks.items():
            print(f"\nChunk {chunk_id} - {chunk['title']}")
            print(f"Summary: {chunk['summary']}")
            print("Propositions:")
            for prop in chunk['propositions']:
                print(f"- {prop}")

    def get_chunks(self, get_type='dict'):
        """Get chunks in specified format"""
        if get_type == 'dict':
            return self.chunks
        elif get_type == 'list_of_strings':
            return ["\n".join(chunk['propositions']) for chunk in self.chunks.values()]
        else:
            raise ValueError("Invalid get_type. Use 'dict' or 'list_of_strings'")

# Main driver
if __name__ == "__main__":
    ac = AgenticChunker()

    propositions = [
        'The month is October.',
        'The year is 2023.',
        "One of the most important things I didn't understand about the world as a child was the degree to which the returns for performance are superlinear.",
        'Teachers and coaches implicitly told us that the returns were linear.',
    ]

    ac.add_propositions(propositions)
    ac.pretty_print_chunks()
    ac.pretty_print_chunk_outline()
    print(ac.get_chunks(get_type='list_of_strings'))