import sys, re
from collections import Counter, defaultdict
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10
plt.style.use('default')

# --------------------------------------------------------------------------
# PATHS
# --------------------------------------------------------------------------
BASE_DIR = r"C:\Users\Lenovo\Desktop\Blockchain & Copyright - Bibliometric Analysis"
papers_path = str(Path(BASE_DIR) / "bibliographic_data" / "final_deduplicated_dataset.csv")
taxonomy_xlsx_path = str(Path(BASE_DIR) / "Industry_categories.xlsx")
out_dir = Path(BASE_DIR) / "analysis_results"
out_dir.mkdir(parents=True, exist_ok=True)

print("BLOCKCHAIN COPYRIGHT INDUSTRY ANALYSIS")
print("="*40)

# --------------------------------------------------------------------------
# LOAD DATA
# --------------------------------------------------------------------------
def read_csv_any(path):
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding='latin1')

print("\nLoading data...")
try:
    original_papers_df = read_csv_any(papers_path)
    print(f"Loaded {len(original_papers_df)} papers from dataset")
except Exception as e:
    print(f"Error loading papers: {e}")
    sys.exit(1)

# Clean data
papers_df = original_papers_df.copy()
for col in ['title', 'keywords', 'abstract', 'authors', 'journal']:
    if col not in papers_df.columns:
        papers_df[col] = ""

papers_df['title'] = papers_df['title'].fillna("").astype(str)
papers_df['keywords'] = papers_df['keywords'].fillna("").astype(str)
papers_df['abstract'] = papers_df['abstract'].fillna("").astype(str)

if 'year' not in papers_df.columns:
    papers_df['year'] = np.nan
papers_df['year'] = pd.to_numeric(papers_df['year'], errors='coerce')

try:
    taxonomy_df = pd.read_excel(taxonomy_xlsx_path, engine="openpyxl")
    print(f"Loaded taxonomy with {len(taxonomy_df)} keyword mappings")
except Exception as e:
    print(f"Error loading taxonomy: {e}")
    sys.exit(1)

# --------------------------------------------------------------------------
# INDUSTRY CLASSIFICATION SETUP
# --------------------------------------------------------------------------
SELECTED_CATEGORIES = [
    "Music & Audio",
    "Creative Arts", 
    "Culture & Museums",
    "Media & Entertainment",
    "Publishing & Libraries",
    "Education & Training",
    "NFT & Digital Assets"
]

# Industry keyword dictionary
filtered_taxonomy = taxonomy_df[taxonomy_df['Category'].isin(SELECTED_CATEGORIES)].copy()
filtered_taxonomy['Keyword'] = filtered_taxonomy['Keyword'].astype(str).str.strip().str.lower()

CREATIVE_INDUSTRIES = defaultdict(dict)
for _, row in filtered_taxonomy.iterrows():
    keyword = row['Keyword']
    category = row['Category']
    # Weight multi-word phrases higher
    weight = 3 if (" " in keyword or "-" in keyword) else 2
    CREATIVE_INDUSTRIES[category][keyword] = weight

print(f"\nIndustry categories configured:")
for industry, keywords in CREATIVE_INDUSTRIES.items():
    print(f"  • {industry}: {len(keywords)} keywords")

# --------------------------------------------------------------------------
# CLASSIFICATION FUNCTIONS
# --------------------------------------------------------------------------
FIELD_WEIGHTS = {'title': 3.0, 'keywords': 2.0, 'abstract': 1.0}
MIN_ABS = 5.0
MIN_REL = 0.25

def score_text(text_fields, keyword_weights):
    total_score = 0.0
    matched_terms = Counter()
    
    for field, text in text_fields.items():
        if not text:
            continue
        field_weight = FIELD_WEIGHTS.get(field, 1.0)
        
        for keyword, keyword_weight in keyword_weights.items():
            # Use word boundaries for precise matching
            pattern = re.compile(rf'\b{re.escape(keyword)}\b', flags=re.IGNORECASE)
            matches = len(pattern.findall(text))
            if matches > 0:
                total_score += matches * keyword_weight * field_weight
                matched_terms[keyword] += matches
    
    return total_score, matched_terms

def classify_paper(row):
    text_fields = {
        'title': row['title'],
        'keywords': row['keywords'], 
        'abstract': row['abstract']
    }
    
    # Score each industry
    industry_scores = {}
    all_matches = Counter()
    
    for industry, keywords in CREATIVE_INDUSTRIES.items():
        score, matches = score_text(text_fields, keywords)
        industry_scores[industry] = score
        all_matches.update(matches)
    
    # Find best industry match
    if not industry_scores or max(industry_scores.values()) <= 0:
        return {
            'industry': 'Other/Unclassified',
            'confidence': 0.0,
            'matched_terms': '',
            'match_count': 0
        }
    
    # Get top two scores for relative margin calculation
    sorted_scores = sorted(industry_scores.values(), reverse=True)
    top_score = sorted_scores[0]
    second_score = sorted_scores[1] if len(sorted_scores) > 1 else 0.0
    relative_margin = (top_score - second_score) / top_score if top_score > 0 else 0.0
    
    best_industry = max(industry_scores, key=industry_scores.get)
    
    # Apply dual threshold: absolute score >= 5.0 OR relative margin >= 0.25
    if top_score < MIN_ABS and relative_margin < MIN_REL:
        return {
            'industry': 'Other/Unclassified',
            'confidence': 0.0,
            'matched_terms': '',
            'match_count': 0
        }
    
    # Format matched terms for reporting
    top_matches = all_matches.most_common(5)
    matched_terms = ', '.join([f"{term}({count})" for term, count in top_matches])
    
    return {
        'industry': best_industry,
        'confidence': round(float(top_score), 1),
        'matched_terms': matched_terms,
        'match_count': sum(all_matches.values())
    }

# --------------------------------------------------------------------------
# CLASSIFICATION EXECUTION
# --------------------------------------------------------------------------
print("\nClassifying papers...")
classification_results = papers_df.apply(classify_paper, axis=1)
results_df = pd.DataFrame(list(classification_results))

# Merge results with original data
classified_papers = pd.concat([papers_df.reset_index(drop=True), results_df.reset_index(drop=True)], axis=1)

# Filter to industry-specific papers only
industry_papers = classified_papers[classified_papers['industry'] != 'Other/Unclassified'].copy()

# Basic statistics
total_papers = len(original_papers_df)
classified_papers_count = len(industry_papers)
classification_rate = (classified_papers_count / total_papers) * 100

print(f"Classification Results:")
print(f"  Total papers in dataset: {total_papers:,}")
print(f"  Industry-classified papers: {classified_papers_count:,}")
print(f"  Classification rate: {classification_rate:.1f}%")

# --------------------------------------------------------------------------
# ANALYSIS
# --------------------------------------------------------------------------
print("\nAnalyzing industry distribution...")

# Industry distribution
industry_distribution = industry_papers['industry'].value_counts()
print(f"\nPapers by industry:")
for industry, count in industry_distribution.items():
    percentage = (count / classified_papers_count) * 100
    print(f"  • {industry}: {count} papers ({percentage:.1f}%)")

# --------------------------------------------------------------------------
# VISUALIZATIONS
# --------------------------------------------------------------------------
print("\nGenerating visualizations...")

# 1. Industry distribution bar chart
try:
    plt.figure(figsize=(12, max(6, 0.5 * len(industry_distribution))))
    colors = plt.cm.Set3(np.linspace(0, 1, len(industry_distribution)))
    
    y_pos = np.arange(len(industry_distribution))
    plt.barh(y_pos, industry_distribution.values, color=colors, alpha=0.8)
    plt.yticks(y_pos, industry_distribution.index)
    plt.xlabel('Number of Papers', fontweight='bold')
    plt.title('Blockchain Copyright Research by Creative Industry', fontweight='bold', pad=20)
    
    # Add value labels
    for i, count in enumerate(industry_distribution.values):
        plt.text(count + max(industry_distribution.values)*0.01, i, str(count), 
                va='center', fontweight='bold')
    
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / 'industry_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: industry_distribution.png")
except Exception as e:
    print(f"Error creating industry chart: {e}")

# 2. Temporal analysis per industry
try:
    # Filter papers with valid years
    temporal_data = industry_papers[industry_papers['year'].notna()].copy()
    
    if len(temporal_data) > 0:
        yearly_industry = temporal_data.groupby(['year', 'industry']).size().unstack(fill_value=0)
        
        if not yearly_industry.empty:
            fig, ax = plt.subplots(figsize=(14, 8))
            yearly_industry.plot(kind='line', marker='o', linewidth=2.5, markersize=6, 
                               alpha=0.8, ax=ax)
            ax.set_xlabel('Year', weight='bold', fontsize=12)
            ax.set_ylabel('Number of Papers', weight='bold', fontsize=12)
            ax.set_title('Temporal Evolution of Blockchain Copyright Research by Industry', 
                        weight='bold', fontsize=14, pad=20)
            ax.legend(fontsize=8, loc='upper left')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Format x-axis
            year_min, year_max = yearly_industry.index.min(), yearly_industry.index.max()
            if year_max > year_min:
                step = max(1, int((year_max - year_min) / 8))
                ax.set_xticks(range(int(year_min), int(year_max) + 1, step))
            
            plt.tight_layout()
            plt.savefig(out_dir / 'temporal_industry_evolution.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ Saved: temporal_industry_evolution.png")
    else:
        print("! No temporal data available for visualization")
        
except Exception as e:
    print(f"Error creating temporal chart: {e}")

# --------------------------------------------------------------------------
# EXPORT RESULTS
# --------------------------------------------------------------------------
print("\nExporting results...")

# Export classified papers
try:
    export_columns = ['title', 'authors', 'year', 'journal', 'industry', 'confidence', 
                     'matched_terms', 'match_count']
    available_columns = [col for col in export_columns if col in industry_papers.columns]
    
    industry_papers[available_columns].to_csv(
        out_dir / 'blockchain_copyright_industries.csv', 
        index=False, encoding='utf-8'
    )
    print("✓ blockchain_copyright_industries.csv")
except Exception as e:
    print(f"Error exporting main results: {e}")

# Export industry analysis
try:
    industry_analysis = []
    for industry, count in industry_distribution.items():
        subset = industry_papers[industry_papers['industry'] == industry]
        
        # Calculate growth rate (2020+ vs pre-2020)
        recent_papers = (subset['year'] >= 2020).sum() if subset['year'].notna().any() else 0
        older_papers = (subset['year'] < 2020).sum() if subset['year'].notna().any() else 0
        growth_rate = recent_papers / older_papers if older_papers > 0 else float('inf') if recent_papers > 0 else 0
        
        industry_analysis.append({
            'Industry': industry,
            'Paper_Count': count,
            'Percentage': f'{(count/classified_papers_count)*100:.1f}%',
            'Avg_Confidence': f'{subset["confidence"].mean():.1f}',
            'Median_Year': subset['year'].median() if subset['year'].notna().any() else 'N/A',
            'Recent_Papers_2020+': recent_papers,
            'Pre_2020_Papers': older_papers,
            'Growth_Rate_2020+': f'{growth_rate:.1f}x' if growth_rate != float('inf') else 'New field'
        })
    
    pd.DataFrame(industry_analysis).to_csv(
        out_dir / 'industry_summary.csv', 
        index=False, encoding='utf-8'
    )
    print("✓ industry_summary.csv")
except Exception as e:
    print(f"Error exporting industry analysis: {e}")

# Create Excel summary
try:
    with pd.ExcelWriter(out_dir / 'analysis_summary.xlsx', engine='openpyxl') as writer:
        pd.DataFrame(industry_analysis).to_excel(writer, sheet_name='Industry_Analysis', index=False)
        
        # Add temporal data if available
        if 'yearly_industry' in locals() and not yearly_industry.empty:
            yearly_industry.to_excel(writer, sheet_name='Temporal_Analysis')
        
    print("✓ analysis_summary.xlsx")
    
except Exception as e:
    print(f"Error exporting Excel summary: {e}")

# --------------------------------------------------------------------------
# SUMMARY
# --------------------------------------------------------------------------
print(f"\n" + "="*50)
print("ANALYSIS SUMMARY")
print("="*50)
print(f"Dataset: {total_papers:,} total papers")
print(f"Classified: {classified_papers_count:,} papers ({classification_rate:.1f}%)")
print(f"Industries represented: {len(industry_distribution)}")
print(f"Top industry: {industry_distribution.index[0]} ({industry_distribution.iloc[0]} papers)")

if len(industry_papers) > 0:
    print(f"Average confidence: {industry_papers['confidence'].mean():.1f}")
    recent_total = (industry_papers['year'] >= 2020).sum() if industry_papers['year'].notna().any() else 0
    print(f"Recent papers (2020+): {recent_total} ({recent_total/classified_papers_count*100:.1f}%)")

print(f"\nFiles generated: {len(list(out_dir.glob('*.png')) + list(out_dir.glob('*.csv')) + list(out_dir.glob('*.xlsx')))}")
print("="*50)
