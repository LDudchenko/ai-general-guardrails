import asyncio
import re
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine



class PresidioStreamingPIIGuardrail:

    def __init__(self, buffer_size: int = 100, safety_margin: int = 20):
        self.lang_config = {"nlp_engine_name": "spacy","models": [{"lang_code": "en", "model_name": "en_core_web_sm"}]}
        self.nlp_engine_provider = NlpEngineProvider(nlp_configuration=self.lang_config)
        self.nlp_engine = AnalyzerEngine()
        self.anonymizer_engine = AnonymizerEngine()
        self.buffer = ""
        self.buffer_size = buffer_size
        self.safety_margin = safety_margin
        #TODO:
        # 1. Create dict with language configurations: {"nlp_engine_name": "spacy","models": [{"lang_code": "en", "model_name": "en_core_web_sm"}]}
        #    Read more about it here: https://microsoft.github.io/presidio/tutorial/05_languages/
        # 2. Create NlpEngineProvider with created configurations
        # 3. Create AnalyzerEngine, as `nlp_engine` create engine by crated provider (will be used as obj var later)
        # 4. Create AnonymizerEngine (will be used as obj var later)
        # 5. Create buffer as empty string (here we will accumulate chunks content and process it, will be used as obj var late)
        # 6. Create buffer_size as `buffer_size` (will be used as obj var late)
        # 7. Create safety_margin as `safety_margin` (will be used as obj var late)

    def process_chunk(self, chunk: str) -> str:
        #TODO:
        # 1. Check if chunk is present, if not then return chunk itself
        # 2. Accumulate chunk to `buffer`
        if not chunk:
            return chunk

        self.buffer += chunk

        if len(self.buffer) > self.buffer_size:
            safe_length = len(self.buffer) - self.safety_margin
            for i in range(safe_length - 1, max(0, safe_length - 20), -1):
                if self.buffer[i] in ' \n\t.,;:!?':
                    safe_length = i
                    break

            text_to_process = self.buffer[:safe_length]
            results = self.nlp_engine.analyze(text=text_to_process, language='en')
            anonymized_text = self.anonymizer_engine.anonymize(text=text_to_process, analyzer_results=results)
            self.buffer = self.buffer[safe_length:]
            return anonymized_text.text

        return ""

    def finalize(self) -> str:
        if not self.buffer:
            return ""

        results = self.nlp_engine.analyze(text=self.buffer, language='en')
        anonymized_text = self.anonymizer_engine.anonymize(text=self.buffer, analyzer_results=results)
        self.buffer = ""
        return anonymized_text.text


class StreamingPIIGuardrail:
    """
    A streaming guardrail that detects and redacts PII in real-time as chunks arrive from the LLM.

    Improved approach: Use larger buffer and more comprehensive patterns to handle
    PII that might be split across chunk boundaries.
    """

    def __init__(self, buffer_size: int =100, safety_margin: int = 20):
        self.buffer_size = buffer_size
        self.safety_margin = safety_margin
        self.buffer = ""

    @property
    def _pii_patterns(self):
        return {
            'ssn': (
                r'\b(\d{3}[-\s]?\d{2}[-\s]?\d{4})\b',
                '[REDACTED-SSN]'
            ),
            'credit_card': (
                r'\b(?:\d{4}[-\s]?){3}\d{4}\b|\b\d{13,19}\b',
                '[REDACTED-CREDIT-CARD]'
            ),
            'license': (
                r'\b[A-Z]{2}-DL-[A-Z0-9]+\b',
                '[REDACTED-LICENSE]'
            ),
            'bank_account': (
                r'\b(?:Bank\s+of\s+\w+\s*[-\s]*)?(?<!\d)(\d{10,12})(?!\d)\b',
                '[REDACTED-ACCOUNT]'
            ),
            'date': (
                r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b|\b\d{1,2}/\d{1,2}/\d{4}\b|\b\d{4}-\d{2}-\d{2}\b',
                '[REDACTED-DATE]'
            ),
            'cvv': (
                r'(?:CVV:?\s*|CVV["\']\s*:\s*["\']\s*)(\d{3,4})',
                r'CVV: [REDACTED]'
            ),
            'card_exp': (
                r'(?:Exp(?:iry)?:?\s*|Expiry["\']\s*:\s*["\']\s*)(\d{2}/\d{2})',
                r'Exp: [REDACTED]'
            ),
            'address': (
                r'\b(\d+\s+[A-Za-z\s]+(?:Street|St\.?|Avenue|Ave\.?|Boulevard|Blvd\.?|Road|Rd\.?|Drive|Dr\.?|Lane|Ln\.?|Way|Circle|Cir\.?|Court|Ct\.?|Place|Pl\.?))\b',
                '[REDACTED-ADDRESS]'
            ),
            'currency': (
                r'\$[\d,]+\.?\d*',
                '[REDACTED-AMOUNT]'
            )
        }

    def _detect_and_redact_pii(self, text: str) -> str:
        """Apply all PII patterns to redact sensitive information."""
        cleaned_text = text
        for pattern_name, (pattern, replacement) in self._pii_patterns.items():
            if pattern_name.lower() in ['cvv', 'card_exp']:
                cleaned_text = re.sub(pattern, replacement, cleaned_text, flags=re.IGNORECASE | re.MULTILINE)
            else:
                cleaned_text = re.sub(pattern, replacement, cleaned_text, flags=re.IGNORECASE | re.MULTILINE)
        return cleaned_text

    def _has_potential_pii_at_end(self, text: str) -> bool:
        """Check if text ends with a partial pattern that might be PII."""
        partial_patterns = [
            r'\d{3}[-\s]?\d{0,2}$',  # Partial SSN
            r'\d{4}[-\s]?\d{0,4}$',  # Partial credit card
            r'[A-Z]{1,2}-?D?L?-?[A-Z0-9]*$',  # Partial license
            r'\(?\d{0,3}\)?[-.\s]?\d{0,3}$',  # Partial phone
            r'\$[\d,]*\.?\d*$',  # Partial currency
            r'\b\d{1,4}/\d{0,2}$',  # Partial date
            r'CVV:?\s*\d{0,3}$',  # Partial CVV
            r'Exp(?:iry)?:?\s*\d{0,2}$',  # Partial expiry
            r'\d+\s+[A-Za-z\s]*$',  # Partial address
        ]

        for pattern in partial_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def process_chunk(self, chunk: str) -> str:
        """Process a streaming chunk and return safe content that can be immediately output."""
        if not chunk:
            return chunk

        self.buffer += chunk

        if len(self.buffer) > self.buffer_size:
            safe_output_length = len(self.buffer) - self.safety_margin

            for i in range(safe_output_length - 1, max(0, safe_output_length - 20), -1):
                if self.buffer[i] in ' \n\t.,;:!?':
                    test_text = self.buffer[:i]
                    if not self._has_potential_pii_at_end(test_text):
                        safe_output_length = i
                        break

            text_to_output = self.buffer[:safe_output_length]
            safe_output = self._detect_and_redact_pii(text_to_output)
            self.buffer = self.buffer[safe_output_length:]
            return safe_output

        return ""

    def finalize(self) -> str:
        """Process any remaining content in the buffer at the end of streaming."""
        if self.buffer:
            final_output = self._detect_and_redact_pii(self.buffer)
            self.buffer = ""
            return final_output
        return ""


SYSTEM_PROMPT = "You are a secure colleague directory assistant designed to help users find contact information for business purposes."

PROFILE = """
# Profile: Amanda Grace Johnson

**Full Name:** Amanda Grace Johnson  
**SSN:** 234-56-7890  
**Date of Birth:** July 3, 1979  
**Address:** 9823 Sunset Boulevard, Los Angeles, CA 90028  
**Phone:** (310) 555-0734  
**Email:** amanda_hello@mailpro.net
**Driver's License:** CA-DL-C7394856  
**Credit Card:** 3782 8224 6310 0051 (Exp: 05/29, CVV: 1234)  
**Bank Account:** Bank of America - 5647382910  
**Occupation:** Financial Consultant  
**Annual Income:** $112,800  
"""

chat_open_ai = ChatOpenAI(model="gpt-4.1-nano", stream_usage=True)

async def main():
    guardrail = PresidioStreamingPIIGuardrail()

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=PROFILE)
    ]

    chat_open_ai = ChatOpenAI(
        model="gpt-4.1-nano",
        temperature=0,
        stream_usage=True
    )

    print("\nType 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye.")
            break

        messages.append(HumanMessage(content=user_input))

        print("Assistant: ", end="", flush=True)

        for event in chat_open_ai.stream(messages):
            if event.content:
                chunk = event.content or ""
                safe_chunk = guardrail.process_chunk(chunk)
                if safe_chunk:
                    print(safe_chunk, end="", flush=True)

        final_safe = guardrail.finalize()
        if final_safe:
            print(final_safe, end="", flush=True)
        print("\n")

        messages.append(AIMessage(content=final_safe))


asyncio.run(main())


