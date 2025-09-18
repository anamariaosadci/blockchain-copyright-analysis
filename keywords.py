import pandas as pd
import numpy as np
import re
from collections import Counter
from pathlib import Path
import string

def load_and_preprocess_data(file_path):
    """Load CSV file with encoding fallback"""
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(file_path, encoding='latin-1')
        except:
            df = pd.read_csv(file_path, encoding='cp1252')
    
    print(f"Dataset loaded: {len(df)} records")
    return df

def clean_text(text):
    """Clean text for keyword extraction"""
    if pd.isna(text) or text == '':
        return ''
    
    text = str(text).lower()
    
    # Preserve important technical terms and abbreviations
    text = re.sub(r'[\[\]{}]', '', text)
    
    # Keep periods in abbreviations and version numbers
    text = re.sub(r'([a-z])\.([a-z])', r'\1DOTPLACEHOLDER\2', text)
    text = re.sub(r'(\d)\.(\d)', r'\1DOTPLACEHOLDER\2', text)
    
    # Remove other punctuation except hyphens and parentheses
    text = re.sub(r'[^\w\s\-\(\)DOTPLACEHOLDER]', ' ', text)
    
    # Restore preserved dots
    text = text.replace('DOTPLACEHOLDER', '.')
    
    # Clean multiple spaces and normalize hyphens
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'-+', '-', text)
    
    # Clean whitespace around parentheses and hyphens
    text = re.sub(r'\s*\(\s*', ' (', text)
    text = re.sub(r'\s*\)\s*', ') ', text)
    text = re.sub(r'\s+-\s+', ' ', text)
    text = re.sub(r'^-\s*|[\s-]+$', '', text)
    
    return text.strip()

def extract_author_keywords(df, keyword_column='keywords'):
    """Extract keywords from author keyword field"""
    all_keywords = []
    keyword_frequency = Counter()
    
    for idx, row in df.iterrows():
        if pd.notna(row[keyword_column]) and str(row[keyword_column]).strip():
            keyword_string = str(row[keyword_column])
            
            # Split by multiple delimiters
            keywords = re.split(r'[;,\n\r\|]+', keyword_string)
            
            for keyword in keywords:
                processed_keyword = clean_text(keyword)
                
                if processed_keyword and len(processed_keyword.strip()) > 1:
                    clean_keyword = processed_keyword.strip()
                    
                    # Filter out very short or purely numeric keywords
                    if (len(clean_keyword) >= 2 and 
                        not clean_keyword.isdigit() and 
                        not clean_keyword.replace('.', '').replace('-', '').isdigit()):
                        
                        all_keywords.append(clean_keyword)
                        keyword_frequency[clean_keyword] += 1
    
    return all_keywords, keyword_frequency

def extract_text_terms(df, title_col='title', abstract_col='abstract', min_length=2):
    """Extract terms from title and abstract fields"""
    
    stopwords = {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he', 
        'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'will', 'with',
        'but', 'or', 'if', 'this', 'they', 'we', 'you', 'have', 'had', 'been', 'their',
        'said', 'each', 'which', 'she', 'do', 'how', 'what', 'up', 'out', 'so', 'no',
        'can', 'could', 'would', 'should', 'may', 'might', 'must', 'shall', 'will',
        'am', 'is', 'are', 'was', 'were', 'being', 'been', 'have', 'has', 'had',
        'do', 'does', 'did', 'doing', 'get', 'got', 'getting', 'make', 'makes', 'made',
        'go', 'goes', 'went', 'going', 'come', 'comes', 'came', 'coming',
        'take', 'takes', 'took', 'taken', 'taking', 'give', 'gives', 'gave', 'given',
        'also', 'then', 'than', 'now', 'here', 'there', 'where', 'when', 'why', 'how',
        'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
        'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just', 'much', 'many',
        'about', 'after', 'before', 'between', 'during', 'through', 'under', 'over',
        'above', 'below', 'down', 'up', 'off', 'away', 'back', 'out', 'again', 'once',
        'new', 'used', 'using', 'based', 'approach', 'method', 'study', 'paper',
        'research', 'article', 'analysis', 'results', 'conclusion', 'introduction',
        'discuss', 'present', 'propose', 'show', 'find', 'found', 'demonstrate'
    }
    
    all_terms = []
    term_frequency = Counter()
    
    for idx, row in df.iterrows():
        combined_text = ""
        
        # Combine title and abstract
        if title_col in df.columns and pd.notna(row[title_col]):
            combined_text += str(row[title_col]) + " "
        
        if abstract_col in df.columns and pd.notna(row[abstract_col]):
            combined_text += str(row[abstract_col])
        
        if combined_text.strip():
            processed_text = clean_text(combined_text)
            
            # Extract parenthetical expressions
            parenthetical_terms = re.findall(r'\b[\w\s]+\([^)]+\)', processed_text)
            for term in parenthetical_terms:
                clean_term = term.strip()
                if len(clean_term) >= min_length:
                    all_terms.append(clean_term)
                    term_frequency[clean_term] += 1
            
            # Extract abbreviations
            abbreviations = re.findall(r'\b[a-z]{2,6}\b', processed_text)
            for abbr in abbreviations:
                if (len(abbr) >= 2 and abbr not in stopwords and 
                    not abbr.isdigit()):
                    all_terms.append(abbr)
                    term_frequency[abbr] += 1
            
            # Extract hyphenated terms
            hyphenated_terms = re.findall(r'\b[\w]+-[\w]+(?:-[\w]+)*\b', processed_text)
            for term in hyphenated_terms:
                if len(term) >= min_length:
                    all_terms.append(term)
                    term_frequency[term] += 1
            
            # Remove parenthetical expressions and hyphens for regular processing
            text_for_phrases = re.sub(r'\([^)]*\)', '', processed_text)
            text_for_phrases = re.sub(r'\b[\w]+-[\w]+(?:-[\w]+)*\b', '', text_for_phrases)
            
            words = text_for_phrases.split()
            
            # Extract n-grams (1-4 words)
            for n in range(1, 5):
                for i in range(len(words) - n + 1):
                    phrase_words = [words[i + j].strip() for j in range(n)]
                    
                    # Filter words
                    if all(len(w) >= min_length and 
                          w not in stopwords and 
                          not w.isdigit() and 
                          w.replace('.', '').replace('-', '').isalnum() and
                          w for w in phrase_words):
                        
                        phrase = " ".join(phrase_words)
                        if len(phrase) >= min_length:
                            all_terms.append(phrase)
                            term_frequency[phrase] += 1
            
            # Extract version patterns
            version_patterns = re.findall(r'\b[\w]+\s+\d+\.\d+\b', processed_text)
            for pattern in version_patterns:
                clean_pattern = pattern.strip()
                if len(clean_pattern) >= min_length:
                    all_terms.append(clean_pattern)
                    term_frequency[clean_pattern] += 1
    
    return all_terms, term_frequency

def sort_by_frequency(term_freq, min_occurrence=1):
    """Sort terms by frequency"""
    sorted_terms = sorted(term_freq.items(), 
                         key=lambda x: (-x[1], x[0].lower()))
    
    filtered_terms = [(term, freq) for term, freq in sorted_terms if freq >= min_occurrence]
    
    return filtered_terms

def save_keyword_list(keywords_data, filename):
    """Save keyword list with frequencies"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            for keyword, count in keywords_data:
                f.write(f"{keyword} ({count})\n")
        
        print(f"Saved: {filename}")
        return True
    except Exception as e:
        print(f"Error saving {filename}: {e}")
        return False

def generate_statistics(all_terms, term_freq):
    """Generate statistics"""
    total_terms = len(all_terms)
    unique_terms = len(term_freq)
    
    freq_distribution = Counter(term_freq.values())
    
    print(f"\nStatistics:")
    print(f"Total term instances: {total_terms}")
    print(f"Unique terms: {unique_terms}")
    print(f"Average occurrences per term: {total_terms/unique_terms:.2f}")
    
    print(f"\nFrequency distribution:")
    for freq in sorted(freq_distribution.keys(), reverse=True)[:10]:
        count = freq_distribution[freq]
        print(f"  {freq} occurrences: {count} terms")

def main():
    """Main keyword extraction function"""
    file_path = r"C:\Users\Lenovo\Desktop\Blockchain & Copyright - Bibliometric Analysis\results\final_deduplicated_dataset.csv"
    
    print("BLOCKCHAIN COPYRIGHT KEYWORD EXTRACTION")
    print("="*50)
    
    try:
        df = load_and_preprocess_data(file_path)
        
        # Author Keywords
        print("\nExtracting author keywords...")
        
        keyword_column = None
        possible_keyword_columns = ['keywords', 'author_keywords', 'keyword', 'Author Keywords']
        
        for col in possible_keyword_columns:
            if col in df.columns:
                keyword_column = col
                break
        
        if keyword_column:
            all_author_keywords, author_keyword_freq = extract_author_keywords(df, keyword_column)
            
            print(f"Keyword column found: '{keyword_column}'")
            generate_statistics(all_author_keywords, author_keyword_freq)
            
            author_keywords = sort_by_frequency(author_keyword_freq, min_occurrence=1)
            
            print(f"\nTop 25 author keywords:")
            for i, (keyword, freq) in enumerate(author_keywords[:25], 1):
                print(f"{i:2d}. {keyword:<50} ({freq:3d})")
        
        # Title/Abstract Terms
        print(f"\nExtracting title/abstract terms...")
        
        title_col = 'title' if 'title' in df.columns else None
        abstract_col = 'abstract' if 'abstract' in df.columns else None
        
        if title_col or abstract_col:
            all_text_terms, text_term_freq = extract_text_terms(
                df, title_col, abstract_col, min_length=2)
            
            generate_statistics(all_text_terms, text_term_freq)
            
            text_terms = sort_by_frequency(text_term_freq, min_occurrence=2)
            
            print(f"\nTop 20 text terms (min. 2 occurrences):")
            for i, (term, freq) in enumerate(text_terms[:20], 1):
                print(f"{i:2d}. {term:<50} ({freq:3d})")
        
        # Save results
        output_dir = "keywords_output"
        
        try:
            Path(output_dir).mkdir(exist_ok=True)
            
            if 'author_keywords' in locals():
                save_keyword_list(author_keywords, f"{output_dir}/author_keywords.txt")
            
            if 'text_terms' in locals():
                save_keyword_list(text_terms, f"{output_dir}/text_terms.txt")
            
            print(f"\nFiles saved to: {output_dir}/")
            
        except Exception as e:
            print(f"Error creating output directory: {e}")
            print("Saving to current directory...")
            if 'author_keywords' in locals():
                save_keyword_list(author_keywords, "author_keywords.txt")
            if 'text_terms' in locals():
                save_keyword_list(text_terms, "text_terms.txt")
        
        # Summary
        print(f"\nSUMMARY:")
        print(f"Papers analyzed: {len(df)}")
        
        if keyword_column:
            papers_with_keywords = df[keyword_column].notna().sum()
            print(f"Papers with keywords: {papers_with_keywords}")
            print(f"Author keyword instances: {len(all_author_keywords)}")
            print(f"Unique author keywords: {len(author_keyword_freq)}")
        
        if title_col or abstract_col:
            print(f"Text term instances: {len(all_text_terms)}")
            print(f"Unique text terms: {len(text_term_freq)}")
        
        print("\nExtraction complete!")
            
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        print("Please ensure the CSV file is in the same directory as this script.")
    except Exception as e:
        print(f"Error during processing: {e}")

if __name__ == "__main__":
    main()
