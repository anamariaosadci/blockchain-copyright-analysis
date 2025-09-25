import pandas as pd
import numpy as np
import re
from collections import Counter
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

def vosviewer_text_processing(text):
    """Clean text for VOSviewer-compatible processing"""
    if pd.isna(text) or text == '':
        return ''
    
    text = str(text).lower()
    
    # Preserve parentheses, hyphens, periods in numbers
    text = re.sub(r'[^\w\s\-\(\)\.]', ' ', text)
    
    # Keep periods between digits, remove others
    text = re.sub(r'\.(?!\d)', ' ', text)
    text = re.sub(r'(?<!\d)\.', ' ', text)
    
    # Clean multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Clean standalone punctuation
    text = re.sub(r'\s-\s', ' ', text)
    text = re.sub(r'^-\s', '', text)
    text = re.sub(r'\s-$', '', text)
    text = re.sub(r'\s\(\s', ' (', text)
    text = re.sub(r'\s\)\s', ') ', text)
    
    return text.strip()

def vosviewer_keyword_extraction(df, keyword_column='keywords'):
    """Extract keywords from author keyword field"""
    all_keywords = []
    keyword_frequency = Counter()
    
    for idx, row in df.iterrows():
        if pd.notna(row[keyword_column]) and str(row[keyword_column]).strip():
            keyword_string = str(row[keyword_column])
            
            # Split by semicolons first, then commas
            keywords = re.split(r'[;,]+', keyword_string)
            
            for keyword in keywords:
                processed_keyword = vosviewer_text_processing(keyword)
                
                if processed_keyword and len(processed_keyword.strip()) > 0:
                    clean_keyword = processed_keyword.strip()
                    
                    if len(clean_keyword) >= 2 and not clean_keyword.isspace():
                        all_keywords.append(clean_keyword)
                        keyword_frequency[clean_keyword] += 1
    
    return all_keywords, keyword_frequency

def vosviewer_title_abstract_extraction(df, title_col='title', abstract_col='abstract', min_length=3):
    """Extract terms from title and abstract fields"""
    
    vos_stopwords = {
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
        'above', 'below', 'down', 'up', 'off', 'away', 'back', 'out', 'again', 'once'
    }
    
    all_terms = []
    term_frequency = Counter()
    
    for idx, row in df.iterrows():
        combined_text = ""
        
        if title_col in df.columns and pd.notna(row[title_col]):
            combined_text += str(row[title_col]) + " "
        
        if abstract_col in df.columns and pd.notna(row[abstract_col]):
            combined_text += str(row[abstract_col])
        
        if combined_text.strip():
            processed_text = vosviewer_text_processing(combined_text)
            
            # Extract parenthetical expressions
            parenthetical_terms = re.findall(r'\b\w+\s+\([^)]+\)', processed_text)
            for term in parenthetical_terms:
                clean_term = term.strip()
                if len(clean_term) >= min_length:
                    all_terms.append(clean_term)
                    term_frequency[clean_term] += 1
            
            # Remove parenthetical expressions for regular processing
            text_without_parens = re.sub(r'\([^)]*\)', '', processed_text)
            
            words = text_without_parens.split()
            
            for i in range(len(words)):
                # Single terms
                word = words[i].strip()
                if (len(word) >= min_length and word not in vos_stopwords and 
                    not word.isdigit() and word):
                    all_terms.append(word)
                    term_frequency[word] += 1
                
                # Two-word phrases
                if i < len(words) - 1:
                    word1, word2 = words[i].strip(), words[i+1].strip()
                    if (len(word1) >= 2 and len(word2) >= 2 and
                        word1 not in vos_stopwords and word2 not in vos_stopwords and
                        word1 and word2):
                        
                        phrase = f"{word1} {word2}"
                        all_terms.append(phrase)
                        term_frequency[phrase] += 1
                
                # Three-word phrases
                if i < len(words) - 2:
                    word1, word2, word3 = words[i].strip(), words[i+1].strip(), words[i+2].strip()
                    if (all(len(w) >= 2 for w in [word1, word2, word3]) and
                        all(w not in vos_stopwords and w for w in [word1, word2, word3])):
                        
                        phrase = f"{word1} {word2} {word3}"
                        all_terms.append(phrase)
                        term_frequency[phrase] += 1
                
                # Four-word phrases
                if i < len(words) - 3:
                    words_slice = [words[i+j].strip() for j in range(4)]
                    if (all(len(w) >= 2 for w in words_slice) and
                        all(w not in vos_stopwords and w for w in words_slice)):
                        
                        phrase = " ".join(words_slice)
                        all_terms.append(phrase)
                        term_frequency[phrase] += 1
            
            # Extract version patterns
            version_patterns = re.findall(r'\b\w+\s+\d+\.\d+\b', processed_text)
            for pattern in version_patterns:
                clean_pattern = pattern.strip()
                if len(clean_pattern) >= min_length:
                    all_terms.append(clean_pattern)
                    term_frequency[clean_pattern] += 1
    
    return all_terms, term_frequency

def create_vosviewer_compatible_output(keyword_freq, min_occurrence=1):
    """Sort and filter keywords by frequency"""
    sorted_keywords = sorted(keyword_freq.items(), 
                           key=lambda x: (-x[1], x[0].lower()))
    
    filtered_keywords = [(kw, freq) for kw, freq in sorted_keywords if freq >= min_occurrence]
    
    return filtered_keywords

def save_vosviewer_format(keywords_data, filename):
    """Save in VOSviewer-compatible tab-separated format"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("id\tlabel\tweight\n")
            
            for i, (keyword, weight) in enumerate(keywords_data, 1):
                clean_keyword = keyword.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ')
                f.write(f"{i}\t{clean_keyword}\t{weight}\n")
        
        print(f"VOSviewer format saved: {filename}")
    except Exception as e:
        print(f"Error saving {filename}: {e}")

def save_keyword_list(keywords_data, filename):
    """Save simple keyword list"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            for keyword, count in keywords_data:
                f.write(f"{keyword}\n")
        
        print(f"Keyword list saved: {filename}")
    except Exception as e:
        print(f"Error saving {filename}: {e}")

def generate_statistics(all_keywords, keyword_freq):
    """Generate keyword extraction statistics"""
    total_keywords = len(all_keywords)
    unique_keywords = len(keyword_freq)
    
    freq_distribution = Counter(keyword_freq.values())
    
    print(f"\nStatistics:")
    print(f"Total instances: {total_keywords}")
    print(f"Unique keywords: {unique_keywords}")
    
    print(f"\nFrequency distribution (top 10):")
    for freq in sorted(freq_distribution.keys(), reverse=True)[:10]:
        count = freq_distribution[freq]
        print(f"  {freq} occurrences: {count} keywords")

def main():
    """Main keyword extraction function"""
    file_path = r"C:\Users\Lenovo\Desktop\Blockchain & Copyright - Bibliometric Analysis\bibliographic_data\final_deduplicated_dataset.csv"
    
    print("KEYWORD EXTRACTION")
    print("="*30)
    
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
            all_author_keywords, author_keyword_freq = vosviewer_keyword_extraction(df, keyword_column)
            
            print(f"Keyword column: '{keyword_column}'")
            print(f"Total instances: {len(all_author_keywords)}")
            print(f"Unique keywords: {len(author_keyword_freq)}")
            
            generate_statistics(all_author_keywords, author_keyword_freq)
            
            vos_author_keywords = create_vosviewer_compatible_output(author_keyword_freq, min_occurrence=1)
            
            print(f"\nTop 20 keywords:")
            for i, (keyword, freq) in enumerate(vos_author_keywords[:20], 1):
                print(f"{i:2d}. {keyword:<40} ({freq:3d})")
        
        else:
            print("No keyword column found")
            return
        
        # Title/Abstract Terms
        print(f"\nExtracting title/abstract terms...")
        
        title_col = 'title' if 'title' in df.columns else None
        abstract_col = 'abstract' if 'abstract' in df.columns else None
        
        if title_col or abstract_col:
            all_text_terms, text_term_freq = vosviewer_title_abstract_extraction(
                df, title_col, abstract_col, min_length=3)
            
            print(f"Total text instances: {len(all_text_terms)}")
            print(f"Unique text terms: {len(text_term_freq)}")
            
            vos_text_terms = create_vosviewer_compatible_output(text_term_freq, min_occurrence=2)
            
            print(f"\nTop 15 text terms (min. 2 occurrences):")
            for i, (term, freq) in enumerate(vos_text_terms[:15], 1):
                print(f"{i:2d}. {term:<40} ({freq:3d})")
        
        # Save results
        output_dir = r"C:\Users\Lenovo\Desktop\Blockchain & Copyright - Bibliometric Analysis\results"
        
        try:
            save_vosviewer_format(vos_author_keywords, f"{output_dir}\\vos_author_keywords.txt")
            save_keyword_list(vos_author_keywords, f"{output_dir}\\author_keywords_list.txt")
            
            if 'vos_text_terms' in locals():
                save_vosviewer_format(vos_text_terms, f"{output_dir}\\vos_text_terms.txt")
                save_keyword_list(vos_text_terms, f"{output_dir}\\text_terms_list.txt")
            
            with open(f"{output_dir}\\keyword_frequencies.txt", 'w', encoding='utf-8') as f:
                f.write("Keyword\tFrequency\n")
                for keyword, freq in vos_author_keywords:
                    f.write(f"{keyword}\t{freq}\n")
            
        except Exception as e:
            print(f"Error saving to output directory: {e}")
            print("Saving to current directory...")
            save_vosviewer_format(vos_author_keywords, "vos_author_keywords.txt")
            save_keyword_list(vos_author_keywords, "author_keywords_list.txt")
        
        # Summary
        print(f"\nSummary:")
        print(f"Papers analyzed: {len(df)}")
        print(f"Papers with keywords: {df[keyword_column].notna().sum()}")
        print(f"Total keyword instances: {len(all_author_keywords)}")
        print(f"Unique keywords extracted: {len(author_keyword_freq)}")
            
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
