import sys, re
from collections import Counter, defaultdict
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

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

# Build industry keyword dictionary
filtered_taxonomy = taxonomy_df[taxonomy_df['Category'].isin(SELECTED_CATEGORIES)].copy()
filtered_taxonomy['Keyword'] = filtered_taxonomy['Keyword'].astype(str).str.strip().str.lower()

CREATIVE_INDUSTRIES = defaultdict(dict)
for _, row in filtered_taxonomy.iterrows():
    keyword = row['Keyword']
    category = row['Category']
    # Multi-word phrases weighted higher
    weight = 3 if (" " in keyword or "-" in keyword) else 2
    CREATIVE_INDUSTRIES[category][keyword] = weight

print(f"\nIndustry categories configured:")
for industry, keywords in CREATIVE_INDUSTRIES.items():
    print(f"  • {industry}: {len(keywords)} keywords")

# --------------------------------------------------------------------------
# CLASSIFICATION FUNCTIONS
# --------------------------------------------------------------------------
# Title and keywords only for precision
FIELD_WEIGHTS = {'title': 3.0, 'keywords': 2.0}

MIN_ABS = 5.0
MIN_REL = 0.25

print(f"\nClassification approach:")
print(f"  • Fields: Title and Keywords only")
print(f"  • Min absolute score: {MIN_ABS}")
print(f"  • Min relative margin: {MIN_REL}")

def score_text(text_fields, keyword_weights):
    total_score = 0.0
    matched_terms = Counter()
    
    for field, text in text_fields.items():
        if not text:
            continue
        field_weight = FIELD_WEIGHTS.get(field, 1.0)
        
        for keyword, keyword_weight in keyword_weights.items():
            # Word boundary matching
            pattern = re.compile(rf'\b{re.escape(keyword)}\b', flags=re.IGNORECASE)
            matches = len(pattern.findall(text))
            if matches > 0:
                total_score += matches * keyword_weight * field_weight
                matched_terms[keyword] += matches
    
    return total_score, matched_terms

def classify_paper(row):
    text_fields = {
        'title': row['title'],
        'keywords': row['keywords']
    }
    
    # Score each industry
    industry_scores = {}
    all_matches = Counter()
    
    for industry, keywords in CREATIVE_INDUSTRIES.items():
        score, matches = score_text(text_fields, keywords)
        industry_scores[industry] = score
        all_matches.update(matches)
    
    if not industry_scores or max(industry_scores.values()) <= 0:
        return {
            'industry': 'Other/Unclassified',
            'confidence': 0.0,
            'matched_terms': '',
            'match_count': 0
        }
    
    # Calculate relative margin
    sorted_scores = sorted(industry_scores.values(), reverse=True)
    top_score = sorted_scores[0]
    second_score = sorted_scores[1] if len(sorted_scores) > 1 else 0.0
    relative_margin = (top_score - second_score) / top_score if top_score > 0 else 0.0
    
    best_industry = max(industry_scores, key=industry_scores.get)
    
    # Dual threshold filter
    if top_score < MIN_ABS and relative_margin < MIN_REL:
        return {
            'industry': 'Other/Unclassified',
            'confidence': 0.0,
            'matched_terms': '',
            'match_count': 0
        }
    
    top_matches = all_matches.most_common(5)
    matched_terms = ', '.join([f"{term}({count})" for term, count in top_matches])
    
    return {
        'industry': best_industry,
        'confidence': round(float(top_score), 1),
        'matched_terms': matched_terms,
        'match_count': sum(all_matches.values()),
        'relative_margin': round(relative_margin, 2)
    }

# --------------------------------------------------------------------------
# CLASSIFICATION EXECUTION
# --------------------------------------------------------------------------
print("\nClassifying papers...")
classification_results = papers_df.apply(classify_paper, axis=1)
results_df = pd.DataFrame(list(classification_results))

classified_papers = pd.concat([papers_df.reset_index(drop=True), results_df.reset_index(drop=True)], axis=1)
industry_papers = classified_papers[classified_papers['industry'] != 'Other/Unclassified'].copy()

# Statistics
total_papers = len(original_papers_df)
classified_papers_count = len(industry_papers)
classification_rate = (classified_papers_count / total_papers) * 100

print(f"\nClassification Results:")
print(f"  Total papers: {total_papers:,}")
print(f"  Industry-classified: {classified_papers_count:,}")
print(f"  Classification rate: {classification_rate:.1f}%")

# --------------------------------------------------------------------------
# ANALYSIS
# --------------------------------------------------------------------------
print("\nAnalyzing industry distribution...")

# Sort industries by count (highest to lowest) - this order will be used everywhere
industry_distribution = industry_papers['industry'].value_counts()
industry_order = industry_distribution.index.tolist()

print(f"\nPapers by industry:")
for industry, count in industry_distribution.items():
    percentage = (count / classified_papers_count) * 100
    avg_conf = industry_papers[industry_papers['industry'] == industry]['confidence'].mean()
    print(f"  • {industry}: {count} papers ({percentage:.1f}%) - Avg confidence: {avg_conf:.1f}")

# --------------------------------------------------------------------------
# VISUALIZATIONS
# --------------------------------------------------------------------------
print("\nGenerating visualizations...")

# Industry distribution bar chart - reverse order for horizontal bar chart
try:
    fig, ax = plt.subplots(figsize=(12, max(6, 0.5 * len(industry_distribution))))
    colors = plt.cm.Set3(np.linspace(0, 1, len(industry_distribution)))
    
    # Reverse the order for display (so highest appears at top)
    industry_display_order = industry_distribution.index[::-1]
    values_display_order = industry_distribution.values[::-1]
    colors_reversed = colors[::-1]
    
    y_pos = np.arange(len(industry_distribution))
    ax.barh(y_pos, values_display_order, color=colors_reversed, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(industry_display_order)
    ax.set_xlabel('Number of Papers', fontweight='bold')
    
    for i, count in enumerate(values_display_order):
        ax.text(count + max(values_display_order)*0.01, i, str(count), 
                va='center', fontweight='bold')
    
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / 'industry_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: industry_distribution.png")
except Exception as e:
    print(f"Error creating industry chart: {e}")

# Temporal analysis
try:
    temporal_data = industry_papers[industry_papers['year'].notna()].copy()
    
    if len(temporal_data) > 0:
        yearly_industry = temporal_data.groupby(['year', 'industry']).size().unstack(fill_value=0)
        
        # Reorder columns by industry_order
        available_industries = [ind for ind in industry_order if ind in yearly_industry.columns]
        yearly_industry = yearly_industry[available_industries]
        
        if not yearly_industry.empty:
            fig, ax = plt.subplots(figsize=(14, 8))
            yearly_industry.plot(kind='line', marker='o', linewidth=2.5, markersize=6, 
                               alpha=0.8, ax=ax)
            ax.set_xlabel('Year', weight='bold', fontsize=12)
            ax.set_ylabel('Number of Papers', weight='bold', fontsize=12)
            ax.legend(fontsize=8, loc='upper left')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            year_min, year_max = yearly_industry.index.min(), yearly_industry.index.max()
            if year_max > year_min:
                step = max(1, int((year_max - year_min) / 8))
                ax.set_xticks(range(int(year_min), int(year_max) + 1, step))
            
            plt.tight_layout()
            plt.savefig(out_dir / 'temporal_industry_evolution.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ Saved: temporal_industry_evolution.png")
    else:
        print("! No temporal data available")
        
except Exception as e:
    print(f"Error creating temporal chart: {e}")

# --------------------------------------------------------------------------
# EXPORT RESULTS
# --------------------------------------------------------------------------
print("\nExporting results...")

# Export classified papers
try:
    export_columns = ['title', 'authors', 'year', 'journal', 'industry', 'confidence', 
                     'relative_margin', 'matched_terms', 'match_count']
    available_columns = [col for col in export_columns if col in industry_papers.columns]
    
    industry_papers[available_columns].to_csv(
        out_dir / 'blockchain_copyright_industries.csv', 
        index=False, encoding='utf-8'
    )
    print("✓ blockchain_copyright_industries.csv")
except Exception as e:
    print(f"Error exporting results: {e}")

# Excel summary - sorted by industry order
try:
    industry_analysis = []
    for industry in industry_order:  # Use the sorted order
        count = industry_distribution[industry]
        subset = industry_papers[industry_papers['industry'] == industry]
        valid_year_subset = subset[subset['year'].notna()]
        
        # Growth rate calculation
        recent_papers = (valid_year_subset['year'] >= 2020).sum() if len(valid_year_subset) > 0 else 0
        older_papers = (valid_year_subset['year'] < 2020).sum() if len(valid_year_subset) > 0 else 0
        
        if older_papers > 0:
            growth_rate = f'{recent_papers / older_papers:.2f}x'
        elif recent_papers > 0:
            growth_rate = 'New field'
        else:
            growth_rate = 'N/A'
        
        if len(valid_year_subset) > 0:
            modal_year = int(valid_year_subset['year'].mode().iloc[0]) if not valid_year_subset['year'].mode().empty else 'N/A'
            first_year = int(valid_year_subset['year'].min())
        else:
            modal_year = 'N/A'
            first_year = 'N/A'
        
        avg_confidence = subset['confidence'].mean()
        avg_margin = subset['relative_margin'].mean()
        
        industry_analysis.append({
            'Industry': industry,
            'Number of Papers': count,
            '% of Total': f'{(count/classified_papers_count)*100:.1f}%',
            'Avg Confidence': f'{avg_confidence:.1f}',
            'Avg Margin': f'{avg_margin:.2f}',
            'Growth Rate': growth_rate,
            'Modal Year': modal_year,
            'First Publication Year': first_year
        })
    
    summary_df = pd.DataFrame(industry_analysis)
    summary_df.to_excel(out_dir / 'industry_summary.xlsx', index=False, engine='openpyxl')
    print("✓ industry_summary.xlsx")
    
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
if len(industry_distribution) > 0:
    print(f"Top industry: {industry_distribution.index[0]} ({industry_distribution.iloc[0]} papers)")

if len(industry_papers) > 0:
    print(f"Average confidence: {industry_papers['confidence'].mean():.1f}")
    print(f"Average margin: {industry_papers['relative_margin'].mean():.2f}")
    recent_total = (industry_papers['year'] >= 2020).sum() if industry_papers['year'].notna().any() else 0
    print(f"Recent papers (2020+): {recent_total} ({recent_total/classified_papers_count*100:.1f}%)")

print(f"\nFiles generated: {len(list(out_dir.glob('*.png')) + list(out_dir.glob('*.csv')) + list(out_dir.glob('*.xlsx')))}")
print(f"\nThresholds: MIN_ABS={MIN_ABS}, MIN_REL={MIN_REL}")
print("="*50)
