
import json
import fitz  #PyMuPDF library
from google import genai
from pydantic import ValidationError

# Import our map to the API keys and our master blueprint
from core.config import settings
from models.candidate import CandidateSchema

# Initialize the Gemini Client safely using our config
client = genai.Client(api_key=settings.GEMINI_API_KEY)

def extract_text_from_pdf(file_path: str) -> str:
    """Reads a PDF file, extracts raw text, and hunts for hidden hyperlinks."""
    try:
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text("text") + "\n"
            
            # 2. Extract hidden hyperlinks (like hyperlinked words)
            links = page.get_links()
            if links:
                text += "\n--- HIDDEN LINKS FOUND ON THIS PAGE ---\n"
                for link in links:
                    if "uri" in link:
                        text += link["uri"] + "\n"
                        
        return text
    except Exception as e:
        print(f"Error reading PDF {file_path}: {e}")
        return ""

def parse_cv_with_gemini(raw_text: str) -> CandidateSchema | None:
    """
    Sends the raw text to Gemini, injecting our Pydantic schema into the prompt 
    to bypass SDK translation errors.
    """
    if not raw_text.strip():
        print("Warning: No text provided to parse.")
        return None

    # We convert our Pydantic model into a raw JSON schema string
    schema_instructions = json.dumps(CandidateSchema.model_json_schema())

    prompt = f"""you are an expert profile  AI crawler  through the linkedin and gethub . 
    your job is mainly to visit linkedin and gethub
    and check if those links are correct
    2.extract inforamtion such as projects and experiences from their linkedin and gethuba.
    

    
    CRITICAL INSTRUCTIONS:
    1. You must return ONLY valid JSON.
    2. Your JSON must strictly adhere to this exact schema structure: 
    {schema_instructions}
    3. If a field is missing from the CV, use null, an empty string "", or an empty list [] based on the schema types.
    
    CV TEXT TO ANALYZE:
    {raw_text}
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                # We keep this to force JSON output, but remove response_schema
                response_mime_type="application/json",
                temperature=0.1, 
            ),
        )
        
        parsed_candidate = CandidateSchema.model_validate_json(response.text)
        return parsed_candidate
        
    except ValidationError as e:
        print(f"Pydantic Validation Error: Gemini returned data in the wrong format.\n{e}")
        return None
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return None

# ==========================================
# Quick Test Block (Only runs if you execute this specific file)
# ==========================================
if __name__ == "__main__":
    print("Agent 01: Ready for testing.")
    # To test this locally without the API:
    # 1. Put a real resume named 'test.pdf' in the root folder
    # 2. Uncomment the lines below
    
    # test_pdf_path = "../test.pdf"
    # extracted_text = extract_text_from_pdf(test_pdf_path)
    # result = parse_cv_with_gemini(extracted_text)
    # if result:
    #     print("\n--- SUCCESS! Extracted JSON: ---")
    #     print(result.model_dump_json(indent=2))
