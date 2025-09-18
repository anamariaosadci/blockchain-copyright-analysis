import pandas as pd
import numpy as np
from pathlib import Path
import re
from difflib import SequenceMatcher
from collections import defaultdict
import rispy

def load_ris_file(file_path, database_name):
    """Load RIS file and convert to DataFrame"""
    print(f"Loading {database_name} data from {file_path}...")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            entries = rispy.load(f)
        
        records = []
        for entry in entries:
            record = {
                'title': entry.get('title', [''])[0] if isinstance(entry.get('title'), list) else entry.get('title', ''),
                'authors': '; '.join(entry.get('authors', [])) if entry.get('authors') else '',
                'year': entry.get('year', ''),
                'journal': entry.get('journal_name', '') or entry.get('secondary_title', ''),
                'abstract': entry.get('abstract', ''),
                'keywords': '; '.join(entry.get('keywords', [])) if entry.get('keywords') else '',
                'doi': entry.get('doi', ''),
                'url': entry.get('url', ''),
                'database': database_name,
                'document_type': entry.get('type_of_reference', ''),
                'volume': entry.get('volume', ''),
                'issue': entry.get('number', ''),
                'pages': entry.get('start_page', '') + '-' + entry.get('end_page', '') if entry.get('start_page') else '',
                'publisher': entry.get('publisher', ''),
                'issn': entry.get('issn', ''),
                'language': entry.get('language', ''),
                'times_cited': entry.get('notes', ''),
                'research_areas': entry.get('research_areas', ''),
                'author_address': '; '.join(entry.get('author_address', [])) if entry.get('author_address') else ''
            }
            records.append(record)
        
        df = pd.DataFrame(records)
        print(f"Loaded {len(df)} records from {database_name}")
        return df
        
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return pd.DataFrame()

def clean_text_for_comparison(text):
    """Clean text for similarity comparison"""
    if pd.isna(text) or text == '':
        return ''
    
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def fast_similarity_check(text1, text2, threshold=0.85):
    """Calculate text similarity with optimization"""
    if not text1 or not text2:
        return 0.0
    
    len1, len2 = len(text1), len(text2)
    if len1 == 0 or len2 == 0:
        return 0.0
    
    length_ratio = min(len1, len2) / max(len1, len2)
    if length_ratio < 0.5:
        return 0.0
    
    clean1 = clean_text_for_comparison(text1)
    clean2 = clean_text_for_comparison(text2)
    
    if clean1 == clean2:
        return 1.0
    
    try:
        return SequenceMatcher(None, clean1, clean2).ratio()
    except:
        return 0.0

def find_duplicates_optimized(df, title_threshold=0.85):
    """Find duplicate papers using DOI and title similarity"""
    print("Searching for duplicates...")
    print(f"Total comparisons needed: ~{len(df)*(len(df)-1)//2:,}")
    
    duplicates = []
    processed_indices = set()
    
    doi_groups = defaultdict(list)
    for idx, row in df.iterrows():
        doi = str(row.get('doi', '')).strip().lower()
        if doi and doi != 'nan' and len(doi) > 5:
            doi_groups[doi].append(idx)
    
    for doi, indices in doi_groups.items():
        if len(indices) > 1:
            duplicates.append({
                'group_id': len(duplicates) + 1,
                'indices': indices,
                'match_reason': 'Identical DOI',
                'papers': df.iloc[indices].to_dict('records')
            })
            processed_indices.update(indices)
    
    print(f"Found {len(duplicates)} DOI duplicate groups")
    print(f"Remaining papers to check: {len(df) - len(processed_indices)}")
    
    remaining_indices = [i for i in range(len(df)) if i not in processed_indices]
    
    total_comparisons = len(remaining_indices) * (len(remaining_indices) - 1) // 2
    completed = 0
    progress_interval = max(1000, total_comparisons // 100)
    
    for i, idx1 in enumerate(remaining_indices):
        if idx1 in processed_indices:
            continue
            
        current_paper = df.iloc[idx1]
        duplicate_group = [idx1]
        
        for j in range(i + 1, len(remaining_indices)):
            idx2 = remaining_indices[j]
            if idx2 in processed_indices:
                continue
                
            completed += 1
            if completed % progress_interval == 0:
                progress = (completed / total_comparisons) * 100
                print(f"   Progress: {progress:.1f}% ({completed:,}/{total_comparisons:,})")
            
            other_paper = df.iloc[idx2]
            
            year1 = str(current_paper.get('year', ''))
            year2 = str(other_paper.get('year', ''))
            if year1 and year2 and year1 != 'nan' and year2 != 'nan':
                try:
                    if abs(int(float(year1)) - int(float(year2))) > 1:
                        continue
                except:
                    pass
            
            title_similarity = fast_similarity_check(
                current_paper.get('title', ''), 
                other_paper.get('title', ''), 
                title_threshold
            )
            
            if title_similarity >= title_threshold:
                duplicate_group.append(idx2)
                processed_indices.add(idx2)
        
        if len(duplicate_group) > 1:
            duplicates.append({
                'group_id': len(duplicates) + 1,
                'indices': duplicate_group,
                'match_reason': f"Title similarity >= {title_threshold}",
                'papers': df.iloc[duplicate_group].to_dict('records')
            })
            
        processed_indices.add(idx1)
    
    total_duplicates = sum(len(group['indices']) - 1 for group in duplicates)
    print(f"Found {len(duplicates)} duplicate groups")
    print(f"Total duplicate records to be removed: {total_duplicates}")
    
    return duplicates

def select_best_record_from_duplicates(df, duplicate_groups):
    """Create deduplicated dataset by selecting best record from each group"""
    print("Creating deduplicated dataset...")
    
    keep_indices = set(range(len(df)))
    removal_summary = defaultdict(int)
    
    database_priority = {
        'Web of Science': 3,
        'Scopus': 2, 
        'OpenAlex': 1
    }
    
    for group in duplicate_groups:
        indices = group['indices']
        
        for idx in indices:
            keep_indices.discard(idx)
            removal_summary[df.iloc[idx]['database']] += 1
        
        best_index = None
        best_priority = -1
        
        for idx in indices:
            db_name = df.iloc[idx]['database']
            priority = database_priority.get(db_name, 0)
            
            record = df.iloc[idx]
            completeness = sum(1 for field in ['title', 'authors', 'year', 'journal', 'abstract', 'doi'] 
                             if record[field] and str(record[field]) != 'nan' and str(record[field]).strip() != '')
            
            total_score = priority * 10 + completeness
            
            if total_score > best_priority:
                best_priority = total_score
                best_index = idx
        
        if best_index is not None:
            keep_indices.add(best_index)
            removal_summary[df.iloc[best_index]['database']] -= 1
    
    final_df = df.iloc[list(keep_indices)].copy().reset_index(drop=True)
    
    print(f"Deduplicated dataset: {len(final_df)} records")
    print("Records removed by database:")
    for db, removed_count in removal_summary.items():
        if removed_count > 0:
            print(f"     {db}: {removed_count} duplicates")
    
    return final_df

def generate_comprehensive_statistics(original_df, final_df, duplicate_groups):
    """Generate statistics about dataset processing"""
    print("Generating statistics...")
    
    overall_stats = {
        'original_total': len(original_df),
        'final_total': len(final_df),
        'duplicates_removed': len(original_df) - len(final_df),
        'duplicate_groups': len(duplicate_groups)
    }
    
    database_stats = {}
    for db in original_df['database'].unique():
        original_count = len(original_df[original_df['database'] == db])
        final_count = len(final_df[final_df['database'] == db])
        database_stats[db] = {
            'original': original_count,
            'final': final_count,
            'removed': original_count - final_count,
            'retention_rate': (final_count / original_count * 100) if original_count > 0 else 0
        }
    
    year_distribution = final_df['year'].value_counts().sort_index()
    top_journals = final_df['journal'].value_counts().head(15)
    doc_types = final_df['document_type'].value_counts()
    
    quality_metrics = {}
    for field in ['title', 'authors', 'year', 'journal', 'abstract', 'doi', 'keywords']:
        filled_count = final_df[field].notna().sum()
        quality_metrics[field] = {
            'filled': int(filled_count),
            'percentage': (filled_count / len(final_df) * 100) if len(final_df) > 0 else 0
        }
    
    return {
        'overall': overall_stats,
        'by_database': database_stats,
        'by_year': year_distribution.to_dict(),
        'top_journals': top_journals.to_dict(),
        'document_types': doc_types.to_dict(),
        'data_quality': quality_metrics
    }

def print_detailed_statistics(stats):
    """Print formatted statistics"""
    print("\n" + "="*50)
    print("DEDUPLICATION RESULTS")
    print("="*50)
    
    print(f"\nOVERALL:")
    print(f"   Original: {stats['overall']['original_total']:,}")
    print(f"   Final: {stats['overall']['final_total']:,}")
    print(f"   Removed: {stats['overall']['duplicates_removed']:,}")
    print(f"   Groups: {stats['overall']['duplicate_groups']:,}")
    
    print(f"\nBY DATABASE:")
    for db, db_stats in stats['by_database'].items():
        print(f"   {db}: {db_stats['final']:,} ({db_stats['retention_rate']:.1f}%)")
    
    print(f"\nDATA QUALITY:")
    for field, metrics in stats['data_quality'].items():
        print(f"   {field.title()}: {metrics['percentage']:.1f}%")

def save_as_ris(df, filename):
    """Save DataFrame as RIS file"""
    print(f"   Converting to RIS...")
    
    ris_entries = []
    
    for _, row in df.iterrows():
        entry = {}
        
        doc_type = str(row.get('document_type', '')).upper()
        if 'ARTICLE' in doc_type or 'JOUR' in doc_type:
            entry['type_of_reference'] = 'JOUR'
        elif 'CONF' in doc_type or 'PROC' in doc_type:
            entry['type_of_reference'] = 'CONF'
        elif 'BOOK' in doc_type:
            entry['type_of_reference'] = 'BOOK'
        elif 'THES' in doc_type:
            entry['type_of_reference'] = 'THES'
        else:
            entry['type_of_reference'] = 'GEN'
        
        if row.get('title') and str(row['title']) != 'nan':
            entry['title'] = str(row['title'])
        
        if row.get('authors') and str(row['authors']) != 'nan':
            authors = [author.strip() for author in str(row['authors']).split(';') if author.strip()]
            if authors:
                entry['authors'] = authors
        
        if row.get('year') and str(row['year']) != 'nan':
            try:
                entry['year'] = str(int(float(row['year'])))
            except (ValueError, TypeError):
                entry['year'] = str(row['year'])
        
        if row.get('journal') and str(row['journal']) != 'nan':
            entry['journal_name'] = str(row['journal'])
        
        if row.get('abstract') and str(row['abstract']) != 'nan':
            entry['abstract'] = str(row['abstract'])
        
        if row.get('keywords') and str(row['keywords']) != 'nan':
            keywords = [kw.strip() for kw in str(row['keywords']).split(';') if kw.strip()]
            if keywords:
                entry['keywords'] = keywords
        
        if row.get('doi') and str(row['doi']) != 'nan':
            entry['doi'] = str(row['doi'])
        
        if row.get('url') and str(row['url']) != 'nan':
            entry['url'] = str(row['url'])
        
        if row.get('volume') and str(row['volume']) != 'nan':
            entry['volume'] = str(row['volume'])
        
        if row.get('issue') and str(row['issue']) != 'nan':
            entry['number'] = str(row['issue'])
        
        if row.get('pages') and str(row['pages']) != 'nan':
            pages = str(row['pages'])
            if '-' in pages:
                start_page, end_page = pages.split('-', 1)
                if start_page.strip():
                    entry['start_page'] = start_page.strip()
                if end_page.strip():
                    entry['end_page'] = end_page.strip()
            else:
                entry['start_page'] = pages
        
        if row.get('publisher') and str(row['publisher']) != 'nan':
            entry['publisher'] = str(row['publisher'])
        
        if row.get('issn') and str(row['issn']) != 'nan':
            entry['issn'] = str(row['issn'])
        
        if row.get('language') and str(row['language']) != 'nan':
            entry['language'] = str(row['language'])
        
        if row.get('database'):
            entry['notes'] = f"Source: {row['database']}"
        
        ris_entries.append(entry)
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            rispy.dump(ris_entries, f)
        print(f"   Saved {len(ris_entries)} records to RIS")
    except Exception as e:
        print(f"   Error saving RIS: {str(e)}")

def save_all_results(final_df, duplicate_groups, stats, output_folder="results"):
    """Save results to files"""
    print(f"\nSaving to '{output_folder}'...")
    
    Path(output_folder).mkdir(exist_ok=True)
    
    final_df.to_csv(f"{output_folder}/final_deduplicated_dataset.csv", index=False, encoding='utf-8')
    print(f"   Final dataset: CSV saved")
    
    save_as_ris(final_df, f"{output_folder}/final_deduplicated_dataset.ris")
    print(f"   Final dataset: RIS saved")
    
    if duplicate_groups:
        duplicate_records = []
        for group in duplicate_groups:
            for i, record in enumerate(group['papers']):
                record['duplicate_group_id'] = group['group_id']
                record['match_reason'] = group['match_reason']
                record['position_in_group'] = i + 1
                record['total_in_group'] = len(group['papers'])
                duplicate_records.append(record)
        
        pd.DataFrame(duplicate_records).to_csv(f"{output_folder}/duplicate_groups_detailed.csv", 
                                              index=False, encoding='utf-8')
        print(f"   Duplicate analysis: saved")
    
    with open(f"{output_folder}/processing_report.txt", 'w', encoding='utf-8') as f:
        f.write("DEDUPLICATION REPORT\n")
        f.write("="*25 + "\n\n")
        
        f.write("RESULTS:\n")
        f.write(f"Original: {stats['overall']['original_total']:,}\n")
        f.write(f"Final: {stats['overall']['final_total']:,}\n")
        f.write(f"Removed: {stats['overall']['duplicates_removed']:,}\n")
        f.write(f"Groups: {stats['overall']['duplicate_groups']:,}\n\n")
        
        f.write("BY DATABASE:\n")
        for db, db_stats in stats['by_database'].items():
            f.write(f"{db}: {db_stats['final']:,} ({db_stats['retention_rate']:.1f}%)\n")
    
    print(f"   Processing report: saved")
    
    analysis_df = final_df[['title', 'authors', 'year', 'journal', 'abstract', 
                           'keywords', 'doi', 'database', 'document_type', 'times_cited']].copy()
    analysis_df.to_csv(f"{output_folder}/dataset_for_bibliometric_analysis.csv", 
                      index=False, encoding='utf-8')
    print(f"   Analysis dataset: saved")

def main():
    """Main processing function"""
    print("BIBLIOMETRIC DATA DEDUPLICATION")
    print("="*35)
    
    ris_files = {
        'Web of Science': 'wos_data.ris',
        'Scopus': 'scopus_data.ris', 
        'OpenAlex': 'openalex_data.ris'
    }
    
    print("\nLoading RIS files...")
    all_datasets = []
    
    for database, filename in ris_files.items():
        if Path(filename).exists():
            df = load_ris_file(filename, database)
            if not df.empty:
                all_datasets.append(df)
                print(f"   {database}: {len(df):,} records")
        else:
            print(f"   Warning: {filename} not found")
    
    if not all_datasets:
        print("\nNo files loaded.")
        return
    
    print(f"\nCombining datasets...")
    combined_data = pd.concat(all_datasets, ignore_index=True)
    print(f"   Total: {len(combined_data):,} records")
    
    print(f"\nFinding duplicates...")
    duplicate_groups = find_duplicates_optimized(combined_data)
    
    print(f"\nCreating final dataset...")
    final_dataset = select_best_record_from_duplicates(combined_data, duplicate_groups)
    
    print(f"\nAnalyzing results...")
    statistics = generate_comprehensive_statistics(combined_data, final_dataset, duplicate_groups)
    
    print_detailed_statistics(statistics)
    
    print(f"\nSaving results...")
    save_all_results(final_dataset, duplicate_groups, statistics)
    
    print(f"\nProcessing complete.")
    print(f"Final dataset: {len(final_dataset):,} unique papers")

if __name__ == "__main__":
    main()