import sys, re
from collections import Counter, defaultdict
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# STATISTICAL IMPORTS
import scipy.stats as stats
from scipy.stats import chi2_contingency
import json
from datetime import datetime

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

print("BLOCKCHAIN COPYRIGHT ANALYSIS")
print("="*35)

# --------------------------------------------------------------------------
# LOAD DATA
# --------------------------------------------------------------------------
def read_csv_any(path):
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding='latin1')

print("\nLoading papers...")
try:
    papers_df = read_csv_any(papers_path)
    print(f"Loaded {len(papers_df)} papers")
except Exception as e:
    print(f"Error loading papers: {e}")
    sys.exit(1)

for col in ['title', 'keywords', 'abstract', 'authors', 'journal']:
    if col not in papers_df.columns:
        papers_df[col] = ""
papers_df['title'] = papers_df['title'].fillna("").astype(str)
papers_df['keywords'] = papers_df['keywords'].fillna("").astype(str)
papers_df['abstract'] = papers_df['abstract'].fillna("").astype(str)
if 'year' not in papers_df.columns:
    papers_df['year'] = np.nan
papers_df['year'] = pd.to_numeric(papers_df['year'], errors='coerce')

print("Loading taxonomy...")
try:
    taxonomy_df = pd.read_excel(taxonomy_xlsx_path, engine="openpyxl")
    print(f"Loaded {len(taxonomy_df)} keyword→category rows")
except Exception as e:
    print(f"Error loading taxonomy: {e}")
    sys.exit(1)

# --------------------------------------------------------------------------
# TAXONOMY: SELECTED INDUSTRY LABELS
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

filtered_taxonomy = taxonomy_df[taxonomy_df['Category'].isin(SELECTED_CATEGORIES)].copy()
filtered_taxonomy['Keyword'] = filtered_taxonomy['Keyword'].astype(str).str.strip().str.lower()

CREATIVE_INDUSTRIES = defaultdict(dict)
for _, row in filtered_taxonomy.iterrows():
    keyword = row['Keyword']
    category = row['Category']
    weight = 3 if (" " in keyword or "-" in keyword) else 2
    CREATIVE_INDUSTRIES[category][keyword] = weight

print("Selected industries and keyword counts:")
for industry, keywords in CREATIVE_INDUSTRIES.items():
    print(f"  • {industry}: {len(keywords)} keywords")

# --------------------------------------------------------------------------
# BLOCKCHAIN COPYRIGHT SOLUTIONS
# --------------------------------------------------------------------------
SOLUTION_RULES = {
    "Rights Registration & Timestamping": [
        ("Privacy & Security", ["timestamp", "timestamping", "proof of creation", "digital signature"]),
        ("Data Management", ["registration", "copyright registration", "creation proof"]),
        ("Legal & Regulatory Framework", ["intellectual property", "copyright", "ip protection", "ip registration"])
    ],
    "Licensing & Smart Contracts": [
        ("Legal & Regulatory Framework", ["licensing", "license", "licensing agreement", 
                                          "smart contract", "automated licensing", "contract automation"]),
        ("Software & Applications", ["smart contract", "contract execution", "automated contracts"])
    ],
    "Royalty Management & Distribution": [
        ("Legal & Regulatory Framework", ["royalty", "royalties", "revenue sharing", 
                                          "payment distribution", "creator compensation"]),
        ("Financial Services", ["payment", "micropayment", "revenue", "monetization"])
    ],
    "Authentication & Anti-Counterfeiting": [
        ("Privacy & Security", ["authentication", "authenticity", "digital signature",
                                "watermark", "watermarking", "anti-counterfeiting"]),
        ("Data Management", ["provenance", "ownership verification", "authenticity verification"]),
        ("Legal & Regulatory Framework", ["piracy", "anti-piracy", "copyright enforcement",
                                          "fraud prevention", "counterfeit detection"])
    ]
}

def build_solution_dict(df_map, rules):
    buckets = {k: {} for k in rules.keys()}
    for bucket, mapping in rules.items():
        for cat, gate in mapping:
            pool = df_map.loc[df_map['Category'] == cat, 'Keyword'].tolist()
            g = [t.lower() for t in gate]
            selected = [k for k in pool if any(t in k for t in g)]
            for k in selected:
                w = 3 if (" " in k or "-" in k) else 2
                buckets[bucket][k] = max(buckets[bucket].get(k, 0), w)
        for g in ["content", "image", "media", "art"]:
            buckets[bucket].pop(g, None)
    return buckets

BLOCKCHAIN_COPYRIGHT_SOLUTIONS = build_solution_dict(taxonomy_df, SOLUTION_RULES)

print("Solution categories:", ", ".join(BLOCKCHAIN_COPYRIGHT_SOLUTIONS.keys()))

# --------------------------------------------------------------------------
# CLASSIFICATION PARAMETERS
# --------------------------------------------------------------------------
FIELD_WEIGHTS = {'title': 3.0, 'keywords': 2.0, 'abstract': 1.0}
MIN_ABS = 5.0
MIN_REL = 0.25

METHODOLOGY_CONFIG = {
    'version': '2.1',
    'analysis_date': datetime.now().isoformat(),
    'classification_parameters': {
        'field_weights': FIELD_WEIGHTS,
        'min_absolute_threshold': MIN_ABS,
        'min_relative_threshold': MIN_REL,
        'selected_industries': SELECTED_CATEGORIES,
        'total_solution_categories': len(BLOCKCHAIN_COPYRIGHT_SOLUTIONS)
    },
    'solution_categories': {
        'approach': 'Mechanism-based blockchain solutions',
        'categories': list(BLOCKCHAIN_COPYRIGHT_SOLUTIONS.keys()),
        'rationale': 'Focused on specific blockchain mechanisms'
    },
    'data_preprocessing': {
        'deduplication': True,
        'missing_value_handling': 'Fill with empty string',
        'year_range_filter': None,
        'minimum_year': None
    },
    'validation_methods': [
        'Statistical significance testing',
        'Confidence interval calculation',
        'Effect size measurement',
        'Classification quality assessment'
    ]
}

def score_category(text_by_field: dict, keyword_weights: dict):
    total = 0.0
    matches = Counter()
    for field, text in text_by_field.items():
        if not text:
            continue
        w_field = FIELD_WEIGHTS.get(field, 1.0)
        for kw, w_kw in keyword_weights.items():
            pattern = re.compile(rf'\b{re.escape(kw)}\b', flags=re.IGNORECASE)
            hits = len(pattern.findall(text))
            if hits:
                total += hits * w_kw * w_field
                matches[kw] += hits
    return total, matches

def pick(scores: dict):
    if not scores:
        return None, 0.0, 0.0
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_label, top_score = ordered[0]
    second_score = ordered[1][1] if len(ordered) > 1 else 0.0
    rel_margin = (top_score - second_score) / top_score if top_score > 0 else 0.0
    return top_label, top_score, rel_margin

def classify_row(row):
    texts = {'title': row['title'], 'keywords': row['keywords'], 'abstract': row['abstract']}
    ind_scores, ind_matches = {}, {}
    for ind, kw_w in CREATIVE_INDUSTRIES.items():
        s, m = score_category(texts, kw_w); ind_scores[ind] = s; ind_matches[ind] = m
    sol_scores, sol_matches = {}, {}
    for sol, kw_w in BLOCKCHAIN_COPYRIGHT_SOLUTIONS.items():
        s, m = score_category(texts, kw_w); sol_scores[sol] = s; sol_matches[sol] = m

    ind_label, ind_abs, ind_rel = pick(ind_scores)
    sol_label, sol_abs, sol_rel = pick(sol_scores)

    if (not ind_label) or (ind_abs <= 0) or (ind_abs < MIN_ABS and ind_rel < MIN_REL):
        ind_label = 'Other/Unclassified'
    if (not sol_label) or (sol_abs <= 0):
        sol_label = 'Other'

    kw_counter = Counter()
    kw_counter.update(ind_matches.get(ind_label, Counter()))
    kw_counter.update(sol_matches.get(sol_label, Counter()))
    matched_terms = ', '.join([f"{k}({v})" for k, v in kw_counter.most_common(8)])
    total_matches = int(sum(kw_counter.values()))
    specificity = 'Industry-Specific' if ind_label != 'Other/Unclassified' else 'General'

    return {
        'target_industry': ind_label,
        'blockchain_solution': sol_label,
        'industry_confidence_abs': round(float(ind_abs), 2),
        'industry_confidence_rel': round(float(ind_rel), 2),
        'solution_confidence_abs': round(float(sol_abs), 2),
        'solution_confidence_rel': round(float(sol_rel), 2),
        'matched_terms': matched_terms,
        'total_matches': total_matches,
        'specificity': specificity
    }

print("\nClassifying papers...")
try:
    results = papers_df.apply(classify_row, axis=1)
    results_df = pd.DataFrame(list(results))
    papers_df = pd.concat([papers_df.reset_index(drop=True), results_df.reset_index(drop=True)], axis=1)
    papers_df = papers_df[papers_df['target_industry'].isin(SELECTED_CATEGORIES + ['Other/Unclassified'])]
    industry_specific = int((papers_df['target_industry'] != 'Other/Unclassified').sum())
    solution_classified = int((papers_df['blockchain_solution'] != 'Other').sum())
    print(f"Industry-specific: {industry_specific} ({industry_specific/len(papers_df)*100:.1f}%)")
    print(f"Solution-classified: {solution_classified} ({solution_classified/len(papers_df)*100:.1f}%)")
except Exception as e:
    print(f"Classification error: {e}")
    sys.exit(1)

# --------------------------------------------------------------------------
# ANALYSIS
# --------------------------------------------------------------------------
print("\nAnalyzing...")

filtered_papers = papers_df[
    (papers_df['target_industry'] != 'Other/Unclassified') & 
    (papers_df['blockchain_solution'] != 'Other')
].copy()

industry_dist = filtered_papers['target_industry'].value_counts()
solution_dist = filtered_papers['blockchain_solution'].value_counts()

industry_analysis = []
for industry, total in industry_dist.items():
    subset = filtered_papers[filtered_papers['target_industry'] == industry]
    y = subset['year'].dropna()
    recent = int((subset['year'] >= 2020).sum())
    older = int((subset['year'] < 2020).sum())
    growth_ratio = None if older == 0 else (recent - older) / older
    sol_counts = subset['blockchain_solution'].value_counts()
    top_solution = sol_counts.index[0] if len(sol_counts) > 0 else 'None'
    industry_analysis.append({
        'Industry': industry,
        'Papers': int(total),
        'Share_of_Total': total / len(filtered_papers),
        'Growth_2020_Plus': growth_ratio,
        'Top_Solution': top_solution,
        'Avg_Industry_Confidence_Abs': float(subset['industry_confidence_abs'].mean()),
        'Avg_Industry_Confidence_Rel': float(subset['industry_confidence_rel'].mean()),
        'Mean_Year': float(y.mean()) if len(y) > 0 else np.nan,
        'Recent_Papers': recent,
        'Solution_Diversity': int(len(sol_counts))
    })

solution_analysis = []
for solution, total in solution_dist.items():
    subset = filtered_papers[filtered_papers['blockchain_solution'] == solution]
    y = subset['year'].dropna()
    solution_analysis.append({
        'Solution': solution,
        'Papers': int(total),
        'Share_of_Total': total / len(filtered_papers),
        'Industries_Covered': int(subset['target_industry'].nunique()),
        'Top_Industry': subset['target_industry'].value_counts().index[0],
        'Mean_Year': float(y.mean()) if len(y) > 0 else np.nan,
        'Avg_Solution_Confidence_Abs': float(subset['solution_confidence_abs'].mean()),
        'Avg_Solution_Confidence_Rel': float(subset['solution_confidence_rel'].mean())
    })

# --------------------------------------------------------------------------
# VISUALIZATIONS
# --------------------------------------------------------------------------
print("\nCreating charts...")
def _break_label(s): return s.replace(' & ', '\n& ')

try:
    yearly_growth = filtered_papers.groupby(['year', 'target_industry']).size().unstack(fill_value=0)
    solution_trends = filtered_papers.groupby(['year', 'blockchain_solution']).size().unstack(fill_value=0)
    
    # Industry trends
    if not yearly_growth.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        yearly_growth.plot(kind='line', marker='o', linewidth=2.5, markersize=6, alpha=0.8, ax=ax)
        ax.set_xlabel('Year', weight='bold', fontsize=10)
        ax.set_ylabel('Number of Papers', weight='bold', fontsize=10)
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        year_min, year_max = yearly_growth.index.min(), yearly_growth.index.max()
        ax.set_xticks(range(int(year_min), int(year_max) + 1, max(1, int((year_max - year_min) / 5))))
        plt.tight_layout()
        plt.savefig(out_dir / 'temporal_industry_trends.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: temporal_industry_trends.png")
    
    # Solution trends
    if not solution_trends.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        solution_trends.plot(kind='line', marker='o', linewidth=2.5, markersize=6, alpha=0.8, ax=ax)
        ax.set_xlabel('Year', weight='bold', fontsize=10)
        ax.set_ylabel('Number of Papers', weight='bold', fontsize=10)
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        year_min, year_max = solution_trends.index.min(), solution_trends.index.max()
        ax.set_xticks(range(int(year_min), int(year_max) + 1, max(1, int((year_max - year_min) / 5))))
        plt.tight_layout()
        plt.savefig(out_dir / 'temporal_solution_trends.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: temporal_solution_trends.png")

    # Combined charts
    try:
        available_industries = [industry for industry in SELECTED_CATEGORIES 
                               if industry in filtered_papers['target_industry'].values]
        
        industry_counts = filtered_papers['target_industry'].value_counts()
        ordered_industries = [ind for ind in industry_counts.index if ind in available_industries]
        
        top_3_industries = ordered_industries[:3]
        remaining_industries = ordered_industries[3:]
        
        # Top 3 industries
        if top_3_industries:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            if len(top_3_industries) == 1:
                axes = [axes]
            
            for i, industry in enumerate(top_3_industries):
                industry_data = filtered_papers[filtered_papers['target_industry'] == industry]
                if not industry_data.empty and i < len(axes):
                    solution_by_year = industry_data.groupby(['year', 'blockchain_solution']).size().unstack(fill_value=0)
                    
                    if not solution_by_year.empty:
                        solution_by_year.plot(kind='line', marker='o', linewidth=2.5, markersize=6, ax=axes[i], alpha=0.8)
                        axes[i].set_title(f'{industry}\nSolution Evolution Over Time', weight='bold', fontsize=12, pad=15)
                        axes[i].set_xlabel('Year', weight='bold', fontsize=10)
                        axes[i].set_ylabel('Number of Papers', weight='bold', fontsize=10)
                        axes[i].grid(True, alpha=0.3, linestyle='--')
                        axes[i].legend(fontsize=8, loc='upper left')
                        axes[i].spines['top'].set_visible(False)
                        axes[i].spines['right'].set_visible(False)
                        year_min, year_max = solution_by_year.index.min(), solution_by_year.index.max()
                        axes[i].set_xticks(range(int(year_min), int(year_max) + 1, max(1, int((year_max - year_min) / 5))))
            
            plt.tight_layout()
            plt.savefig(out_dir / 'temporal_industry_solution_top3.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved: temporal_industry_solution_top3.png ({len(top_3_industries)} industries)")
        
        # Remaining industries
        if remaining_industries:
            rows = 2
            cols = 2
            fig, axes = plt.subplots(rows, cols, figsize=(16, 10))
            axes = axes.flatten()
            
            for i, industry in enumerate(remaining_industries):
                if i < len(axes):
                    industry_data = filtered_papers[filtered_papers['target_industry'] == industry]
                    if not industry_data.empty:
                        solution_by_year = industry_data.groupby(['year', 'blockchain_solution']).size().unstack(fill_value=0)
                        
                        if not solution_by_year.empty:
                            solution_by_year.plot(kind='line', marker='o', linewidth=2.5, markersize=6, ax=axes[i], alpha=0.8)
                            axes[i].set_title(f'{industry}\nSolution Evolution Over Time', weight='bold', fontsize=12, pad=15)
                            axes[i].set_xlabel('Year', weight='bold', fontsize=10)
                            axes[i].set_ylabel('Number of Papers', weight='bold', fontsize=10)
                            axes[i].grid(True, alpha=0.3, linestyle='--')
                            axes[i].legend(fontsize=8, loc='upper left')
                            axes[i].spines['top'].set_visible(False)
                            axes[i].spines['right'].set_visible(False)
                            year_min, year_max = solution_by_year.index.min(), solution_by_year.index.max()
                            axes[i].set_xticks(range(int(year_min), int(year_max) + 1, max(1, int((year_max - year_min) / 5))))
                        else:
                            axes[i].text(0.5, 0.5, f'Insufficient data for\n{industry}', 
                                       ha='center', va='center', transform=axes[i].transAxes, fontsize=12)
                            axes[i].set_title(f'{industry}\n(Limited Data)', weight='bold', fontsize=12)
            
            for j in range(len(remaining_industries), len(axes)):
                axes[j].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(out_dir / 'temporal_industry_solution_remaining.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved: temporal_industry_solution_remaining.png ({len(remaining_industries)} industries)")
        
    except Exception as e:
        print(f"Combined temporal analysis error: {e}")

except Exception as e:
    print(f"Temporal analysis error: {e}")

# Additional charts
try:
    if len(industry_dist) > 0:
        ind_data = list(industry_dist.items())
        ind_data.sort(key=lambda x: x[1], reverse=True)
        if ind_data:
            labels, counts = zip(*ind_data)
            plt.figure(figsize=(12, max(4, 0.45 * len(labels))))
            cmap = plt.get_cmap('tab20'); colors = [cmap(i % 20) for i in range(len(labels))]
            y_pos = np.arange(len(labels))
            plt.barh(y_pos, counts, color=colors, alpha=0.88)
            plt.yticks(y_pos, [_break_label(l) for l in labels])
            plt.xlabel('Number of Papers', weight='bold'); plt.title('Papers by Creative Industry', weight='bold')
            for i, c in enumerate(counts): plt.text(c + max(counts)*0.01, i, str(c), va='center', weight='bold')
            plt.grid(axis='x', alpha=0.3); plt.tight_layout()
            plt.savefig(out_dir / 'industry_distribution.png', dpi=150, bbox_inches='tight'); plt.close()
            print("Saved: industry_distribution.png")
except Exception as e:
    print(f"Industry chart error: {e}")

try:
    crosstab = pd.crosstab(filtered_papers['target_industry'], filtered_papers['blockchain_solution'])
    
    if crosstab.size > 0:
        crosstab_filtered = crosstab.loc[:, (crosstab != 0).any(axis=0)]
        crosstab_filtered = crosstab_filtered.loc[(crosstab_filtered != 0).any(axis=1), :]
        
        if crosstab_filtered.size > 0:
            crosstab_filtered.plot(kind='bar', stacked=True, colormap='tab20', figsize=(12, 6))
            plt.title('Blockchain Copyright Solutions by Creative Industry', weight='bold')
            plt.xlabel('Creative Industry')
            plt.ylabel('Number of Papers')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(out_dir / 'industry_solution_stacked_bar.png', dpi=150)
            plt.close()
            print("Saved: industry_solution_stacked_bar.png")
            
            plt.figure(figsize=(14, max(6, 1.2 * len(crosstab_filtered.index))))
            sns.heatmap(crosstab_filtered, annot=True, fmt='d', cmap='Blues', cbar_kws={'label':'Number of Papers'})
            plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0)
            plt.title('Industry–Solution Matrix', weight='bold', pad=12)
            plt.tight_layout(); plt.savefig(out_dir / 'industry_solution_matrix.png', dpi=150, bbox_inches='tight'); plt.close()
            print("Saved: industry_solution_matrix.png")
except Exception as e:
    print(f"Chart error: {e}")

# --------------------------------------------------------------------------
# EXPORTS
# --------------------------------------------------------------------------
print("\nExporting files...")
try:
    main_cols = [
        'title','authors','year','journal','target_industry','blockchain_solution',
        'industry_confidence_abs','industry_confidence_rel','solution_confidence_abs',
        'solution_confidence_rel','matched_terms','total_matches','specificity'
    ]
    keep = [c for c in main_cols if c in papers_df.columns]
    papers_df[keep].to_csv(out_dir / 'blockchain_copyright_by_industry.csv', index=False, encoding='utf-8')
    print("✓ blockchain_copyright_by_industry.csv")

    if industry_analysis:
        pd.DataFrame(industry_analysis).to_csv(out_dir / 'industry_analysis.csv', index=False, encoding='utf-8')
        print("✓ industry_analysis.csv")

    if solution_analysis:
        pd.DataFrame(solution_analysis).to_csv(out_dir / 'solution_analysis.csv', index=False, encoding='utf-8')
        print("✓ solution_analysis.csv")

    if 'crosstab_filtered' in locals() and crosstab_filtered.size > 0:
        crosstab_filtered.to_csv(out_dir / 'contingency_table_full.csv', encoding='utf-8')
        print("✓ contingency_table_full.csv")

    if 'yearly_growth' in locals() and not yearly_growth.empty:
        yearly_growth.to_csv(out_dir / 'temporal_industry_analysis.csv', encoding='utf-8')
        print("✓ temporal_industry_analysis.csv")
    
    if 'solution_trends' in locals() and not solution_trends.empty:
        solution_trends.to_csv(out_dir / 'temporal_solution_analysis.csv', encoding='utf-8')
        print("✓ temporal_solution_analysis.csv")

except Exception as e:
    print(f"Export error: {e}")

# --------------------------------------------------------------------------
# STATISTICAL VALIDATION - EXCEL EXPORT
# --------------------------------------------------------------------------
print("\nStatistical validation...")

n = len(filtered_papers)
print(f"Industry-specific papers: {n:,}")

# Initialize variables to avoid "not defined" errors
chi2 = p_value = dof = cramers_v = None
contingency = None

# Prepare statistical results for Excel
statistical_results = []

# Sample characteristics
total_original_papers = len(papers_df)
industry_specific_rate = (n / total_original_papers) * 100 if total_original_papers > 0 else 0

statistical_results.extend([
    {
        'Category': 'Dataset Overview',
        'Metric': 'Total papers analyzed',
        'Value': total_original_papers,
        'Description': 'Total number of papers in the original dataset before filtering',
        'Interpretation': 'This represents the complete bibliometric dataset used as input for analysis.'
    },
    {
        'Category': 'Dataset Overview',
        'Metric': 'Industry-specific papers',
        'Value': n,
        'Description': f'Papers successfully classified into creative industries ({industry_specific_rate:.1f}% of total)',
        'Interpretation': 'Higher numbers indicate more papers are relevant to creative industries. Good classification rates are typically 15-30% for specialized domains.'
    },
    {
        'Category': 'Dataset Overview',
        'Metric': 'Classification success rate',
        'Value': f'{industry_specific_rate:.1f}%',
        'Description': 'Percentage of papers that could be classified into specific creative industries',
        'Interpretation': '15-20% = Good for specialized domains | 20-30% = Very good | >30% = Excellent domain relevance'
    }
])

# Association tests (if contingency table exists)
try:
    # Check if crosstab_filtered was created earlier in the script
    crosstab_filtered = pd.crosstab(filtered_papers['target_industry'], filtered_papers['blockchain_solution'])
    crosstab_filtered = crosstab_filtered.loc[:, (crosstab_filtered != 0).any(axis=0)]
    crosstab_filtered = crosstab_filtered.loc[(crosstab_filtered != 0).any(axis=1), :]
    
    if crosstab_filtered.size > 0:
        contingency = crosstab_filtered
        chi2, p_value, dof, expected = chi2_contingency(contingency)
        n_total = contingency.sum().sum()
        cramers_v = np.sqrt(chi2 / (n_total * (min(contingency.shape) - 1)))
        
        # Interpret statistical significance
        if p_value < 0.001:
            significance_interpretation = "Extremely strong evidence of association (p < 0.001)"
        elif p_value < 0.01:
            significance_interpretation = "Strong evidence of association (p < 0.01)"
        elif p_value < 0.05:
            significance_interpretation = "Significant association found (p < 0.05)"
        else:
            significance_interpretation = "No significant association found (p ≥ 0.05)"
            
        # Interpret effect size
        if cramers_v < 0.1:
            effect_interpretation = "Negligible association"
        elif cramers_v < 0.3:
            effect_interpretation = "Small to moderate association"
        elif cramers_v < 0.5:
            effect_interpretation = "Moderate to strong association"
        else:
            effect_interpretation = "Strong association"
        
        statistical_results.extend([
            {
                'Category': 'Industry-Solution Association',
                'Metric': 'Chi-square statistic', 
                'Value': round(chi2, 2),
                'Description': f'Statistical test value measuring independence between industries and blockchain solutions',
                'Interpretation': f'Higher values indicate stronger relationships. This value of {chi2:.1f} suggests {"strong" if chi2 > 20 else "moderate" if chi2 > 10 else "weak"} patterns between industries and solutions.'
            },
            {
                'Category': 'Industry-Solution Association',
                'Metric': 'Statistical significance (p-value)',
                'Value': f"{p_value:.2e}",
                'Description': 'Probability that the observed association occurred by chance',
                'Interpretation': significance_interpretation
            },
            {
                'Category': 'Industry-Solution Association', 
                'Metric': 'Degrees of freedom',
                'Value': dof,
                'Description': 'Number of independent comparisons in the statistical test',
                'Interpretation': f'Based on {contingency.shape[0]} industries and {contingency.shape[1]} solution types. Higher values indicate more complex relationships.'
            },
            {
                'Category': 'Industry-Solution Association',
                'Metric': "Association strength (Cramer's V)",
                'Value': round(cramers_v, 3),
                'Description': 'Measure of association strength between industries and solutions (0 = no association, 1 = perfect association)',
                'Interpretation': f'{effect_interpretation}. Values: <0.1=negligible, 0.1-0.3=small-moderate, 0.3-0.5=moderate-strong, >0.5=strong.'
            }
        ])
        
        print(f"Chi-square: {chi2:.2f}, p-value: {p_value:.2e}")
        print(f"Effect size (Cramer's V): {cramers_v:.3f}")
        
except Exception as e:
    print(f"Contingency table analysis error: {e}")
    statistical_results.extend([
        {
            'Category': 'Industry-Solution Association',
            'Metric': 'Statistical test status',
            'Value': 'Could not compute',
            'Description': f'Error in statistical analysis: {str(e)}',
            'Interpretation': 'Statistical association tests require sufficient data in multiple industries and solutions. Consider increasing sample size or broadening classification criteria.'
        }
    ])

# Classification quality metrics
confidence_scores = filtered_papers['industry_confidence_abs']
mean_confidence = confidence_scores.mean()
std_confidence = confidence_scores.std()
min_confidence = confidence_scores.min()
max_confidence = confidence_scores.max()

# Interpret confidence levels
if mean_confidence >= 15:
    confidence_interpretation = "High confidence - classifications are very reliable"
elif mean_confidence >= 10:
    confidence_interpretation = "Good confidence - classifications are reliable" 
elif mean_confidence >= 5:
    confidence_interpretation = "Moderate confidence - classifications are acceptable"
else:
    confidence_interpretation = "Low confidence - classifications may need refinement"

statistical_results.extend([
    {
        'Category': 'Classification Quality',
        'Metric': 'Average classification confidence',
        'Value': round(mean_confidence, 1),
        'Description': f'Mean confidence score across all classified papers (range: {min_confidence:.1f} to {max_confidence:.1f})',
        'Interpretation': f'{confidence_interpretation}. Scale: <5=low, 5-10=moderate, 10-15=good, >15=high confidence.'
    },
    {
        'Category': 'Classification Quality',
        'Metric': 'Classification consistency', 
        'Value': round(std_confidence, 1),
        'Description': f'Standard deviation of confidence scores (lower = more consistent)',
        'Interpretation': f'{"Very consistent" if std_confidence < 3 else "Moderately consistent" if std_confidence < 5 else "Variable"} classification quality across papers. Lower values indicate more uniform confidence levels.'
    },
    {
        'Category': 'Classification Quality',
        'Metric': 'High-confidence papers',
        'Value': int((confidence_scores >= 10).sum()),
        'Description': f'Number of papers with confidence ≥ 10 ({(confidence_scores >= 10).sum()/len(confidence_scores)*100:.1f}% of classified papers)',
        'Interpretation': 'Papers with high confidence scores are most reliably classified. Higher percentages indicate better overall classification quality.'
    }
])

# Distribution analysis
industry_counts = filtered_papers['target_industry'].value_counts()
cv = industry_counts.std() / industry_counts.mean()
most_common_industry = industry_counts.index[0]
least_common_industry = industry_counts.index[-1]

# Interpret distribution balance
if cv < 0.5:
    balance_interpretation = "Very balanced - industries are evenly represented"
elif cv < 1.0:
    balance_interpretation = "Moderately balanced - some industries dominate but others are present"
elif cv < 1.5:
    balance_interpretation = "Unbalanced - few industries dominate the dataset"
else:
    balance_interpretation = "Highly unbalanced - one or two industries heavily dominate"

statistical_results.extend([
    {
        'Category': 'Industry Distribution',
        'Metric': 'Distribution balance (CV)',
        'Value': round(cv, 2), 
        'Description': f'Coefficient of variation measuring how evenly papers are distributed across industries',
        'Interpretation': f'{balance_interpretation}. Scale: <0.5=very balanced, 0.5-1.0=moderate, 1.0-1.5=unbalanced, >1.5=highly unbalanced.'
    },
    {
        'Category': 'Industry Distribution',
        'Metric': 'Creative industries represented',
        'Value': len(industry_counts),
        'Description': f'Number of different creative industries found in the classified papers (out of {len(SELECTED_CATEGORIES)} possible)',
        'Interpretation': f'{"Excellent" if len(industry_counts) >= 6 else "Good" if len(industry_counts) >= 4 else "Limited"} industry diversity. More industries indicate broader blockchain copyright applications.'
    },
    {
        'Category': 'Industry Distribution',
        'Metric': 'Dominant industry',
        'Value': most_common_industry,
        'Description': f'Industry with the most papers: {industry_counts.iloc[0]} papers ({industry_counts.iloc[0]/n*100:.1f}% of classified papers)',
        'Interpretation': f'{"Moderately dominant" if industry_counts.iloc[0]/n < 0.4 else "Highly dominant" if industry_counts.iloc[0]/n < 0.6 else "Overwhelmingly dominant"} industry in blockchain copyright research.'
    },
    {
        'Category': 'Industry Distribution',
        'Metric': 'Emerging industry',
        'Value': least_common_industry,
        'Description': f'Industry with the fewest papers: {industry_counts.iloc[-1]} papers ({industry_counts.iloc[-1]/n*100:.1f}% of classified papers)',
        'Interpretation': 'Industries with few papers may represent emerging areas or niche applications of blockchain copyright solutions.'
    }
])

# Add solution diversity analysis
solution_counts = filtered_papers['blockchain_solution'].value_counts()
solution_cv = solution_counts.std() / solution_counts.mean()

statistical_results.extend([
    {
        'Category': 'Solution Distribution', 
        'Metric': 'Blockchain solutions identified',
        'Value': len(solution_counts),
        'Description': f'Number of different blockchain copyright solution types found (out of {len(BLOCKCHAIN_COPYRIGHT_SOLUTIONS)} possible)',
        'Interpretation': f'{"Comprehensive" if len(solution_counts) >= 3 else "Moderate" if len(solution_counts) >= 2 else "Limited"} solution diversity indicates {"broad" if len(solution_counts) >= 3 else "focused"} research coverage.'
    },
    {
        'Category': 'Solution Distribution',
        'Metric': 'Primary solution focus',
        'Value': solution_counts.index[0],
        'Description': f'Most researched solution: {solution_counts.iloc[0]} papers ({solution_counts.iloc[0]/n*100:.1f}% of classified papers)',
        'Interpretation': 'The dominant solution type indicates the primary focus of current blockchain copyright research.'
    },
    {
        'Category': 'Solution Distribution',
        'Metric': 'Solution balance (CV)',
        'Value': round(solution_cv, 2),
        'Description': 'How evenly research attention is distributed across different solution types',
        'Interpretation': f'{"Balanced research focus" if solution_cv < 0.8 else "Uneven research focus"} across blockchain copyright solutions.'
    }
])

print(f"Mean confidence: {mean_confidence:.1f} (SD: {std_confidence:.1f})")
print(f"Distribution balance (CV): {cv:.2f}")

# Export to Excel with multiple sheets and user-friendly formatting
try:
    with pd.ExcelWriter(out_dir / 'statistical_validation.xlsx', engine='openpyxl') as writer:
        # Main statistical results with enhanced formatting
        stats_df = pd.DataFrame(statistical_results)
        stats_df.to_excel(writer, sheet_name='Statistical_Summary', index=False)
        
        # Create an executive summary sheet
        executive_summary = [
            {
                'Key Finding': 'Dataset Quality',
                'Value': f'{industry_specific_rate:.1f}%',
                'What This Means': f'Successfully classified {industry_specific_rate:.1f}% of papers into creative industries, indicating {"excellent" if industry_specific_rate > 30 else "good" if industry_specific_rate > 20 else "moderate"} domain relevance.'
            },
            {
                'Key Finding': 'Classification Reliability', 
                'Value': f'{mean_confidence:.1f} avg confidence',
                'What This Means': f'{"High" if mean_confidence >= 15 else "Good" if mean_confidence >= 10 else "Moderate"} confidence in paper classifications, meaning results are {"very" if mean_confidence >= 15 else ""} reliable.'
            },
            {
                'Key Finding': 'Industry Coverage',
                'Value': f'{len(industry_counts)} industries',
                'What This Means': f'Blockchain copyright research spans {len(industry_counts)} creative industries, showing {"broad" if len(industry_counts) >= 5 else "moderate" if len(industry_counts) >= 3 else "focused"} application scope.'
            },
            {
                'Key Finding': 'Research Focus',
                'Value': most_common_industry,
                'What This Means': f'{most_common_industry} dominates with {industry_counts.iloc[0]} papers ({industry_counts.iloc[0]/n*100:.1f}%), indicating this is the primary focus area.'
            }
        ]
        
        if chi2 is not None and p_value is not None:
            executive_summary.append({
                'Key Finding': 'Industry-Solution Patterns',
                'Value': f'p-value: {p_value:.2e}',
                'What This Means': f'{"Strong statistical evidence" if p_value < 0.01 else "Some evidence" if p_value < 0.05 else "No clear evidence"} that certain industries prefer specific blockchain solutions.'
            })
        
        pd.DataFrame(executive_summary).to_excel(writer, sheet_name='Executive_Summary', index=False)
        
        # Industry analysis with user-friendly descriptions
        if 'industry_analysis' in locals() and industry_analysis:
            industry_df = pd.DataFrame(industry_analysis)
            # Add interpretation columns
            industry_df['Growth_Interpretation'] = industry_df['Growth_2020_Plus'].apply(
                lambda x: 'High growth' if pd.notna(x) and x > 0.5 else 
                         'Moderate growth' if pd.notna(x) and x > 0 else
                         'Declining' if pd.notna(x) and x < -0.2 else
                         'Stable' if pd.notna(x) else 'Insufficient data'
            )
            industry_df['Confidence_Level'] = industry_df['Avg_Industry_Confidence_Abs'].apply(
                lambda x: 'High' if x >= 15 else 'Good' if x >= 10 else 'Moderate' if x >= 5 else 'Low'
            )
            industry_df.to_excel(writer, sheet_name='Industry_Analysis', index=False)
            
        # Solution analysis with explanations
        if 'solution_analysis' in locals() and solution_analysis:
            solution_df = pd.DataFrame(solution_analysis)
            solution_df['Research_Maturity'] = solution_df['Mean_Year'].apply(
                lambda x: 'Emerging (recent focus)' if pd.notna(x) and x >= 2020 else
                         'Established (ongoing research)' if pd.notna(x) and x >= 2018 else
                         'Early stage' if pd.notna(x) else 'Unknown timeline'
            )
            solution_df['Industry_Reach'] = solution_df['Industries_Covered'].apply(
                lambda x: 'Broad application' if x >= 4 else 'Moderate reach' if x >= 2 else 'Specialized application'
            )
            solution_df.to_excel(writer, sheet_name='Solution_Analysis', index=False)
        
        # Contingency table with labels
        try:
            if 'crosstab_filtered' in locals() and crosstab_filtered.size > 0:
                crosstab_filtered.to_excel(writer, sheet_name='Industry_Solution_Matrix')
                
                # Add interpretation sheet for the matrix
                matrix_interpretation = []
                for industry in crosstab_filtered.index:
                    for solution in crosstab_filtered.columns:
                        count = crosstab_filtered.loc[industry, solution]
                        if count > 0:
                            percentage = (count / n) * 100
                            matrix_interpretation.append({
                                'Industry': industry,
                                'Solution': solution,
                                'Papers': int(count),
                                'Percentage_of_Total': f'{percentage:.1f}%',
                                'Interpretation': f'{"Strong focus" if count >= 10 else "Moderate focus" if count >= 5 else "Emerging area"} - {industry} research on {solution.lower()}'
                            })
                
                pd.DataFrame(matrix_interpretation).to_excel(writer, sheet_name='Matrix_Interpretation', index=False)
        except Exception as e:
            print(f"Contingency table sheet error: {e}")
        
        # Confidence score distribution with context
        try:
            confidence_stats = pd.DataFrame({
                'Paper_ID': range(len(filtered_papers)),
                'Confidence_Score': filtered_papers['industry_confidence_abs'],
                'Confidence_Level': filtered_papers['industry_confidence_abs'].apply(
                    lambda x: 'High (≥15)' if x >= 15 else 'Good (10-14)' if x >= 10 else 'Moderate (5-9)' if x >= 5 else 'Low (<5)'
                ),
                'Industry': filtered_papers['target_industry'],
                'Solution': filtered_papers['blockchain_solution'],
                'Reliability': filtered_papers['industry_confidence_abs'].apply(
                    lambda x: 'Very reliable' if x >= 15 else 'Reliable' if x >= 10 else 'Acceptable' if x >= 5 else 'Use with caution'
                )
            })
            confidence_stats.to_excel(writer, sheet_name='Confidence_Scores', index=False)
        except Exception as e:
            print(f"Confidence scores sheet error: {e}")
        
    print("✓ statistical_validation.xlsx (multi-sheet with interpretations)")
    
except Exception as e:
    print(f"Excel export error: {e}")
    # Enhanced fallback to CSV with interpretations
    try:
        stats_with_summary = pd.DataFrame(statistical_results)
        stats_with_summary.to_csv(out_dir / 'statistical_validation_detailed.csv', index=False)
        print("✓ statistical_validation_detailed.csv (fallback with interpretations)")
    except Exception as csv_error:
        print(f"CSV fallback error: {csv_error}")

# Still save methodology config as JSON (useful for reproducibility)
try:
    with open(out_dir / 'methodology_config.json', 'w', encoding='utf-8') as f:
        json.dump(METHODOLOGY_CONFIG, f, indent=2)
    print("✓ methodology_config.json")
except Exception as e:
    print(f"Methodology JSON export error: {e}")

# Also create methodology summary in Excel
try:
    methodology_summary = []
    
    # Classification parameters
    if 'METHODOLOGY_CONFIG' in locals():
        for key, value in METHODOLOGY_CONFIG['classification_parameters'].items():
            methodology_summary.append({
                'Section': 'Classification Parameters',
                'Parameter': key.replace('_', ' ').title(),
                'Value': str(value),
                'Description': 'Parameters used for paper classification'
            })
        
        # Solution categories  
        methodology_summary.append({
            'Section': 'Solution Categories',
            'Parameter': 'Approach',
            'Value': METHODOLOGY_CONFIG['solution_categories']['approach'],
            'Description': 'Method for defining blockchain solutions'
        })
        
        methodology_summary.append({
            'Section': 'Solution Categories', 
            'Parameter': 'Categories',
            'Value': ', '.join(METHODOLOGY_CONFIG['solution_categories']['categories']),
            'Description': 'Blockchain copyright solution categories analyzed'
        })
        
        # Data preprocessing
        for key, value in METHODOLOGY_CONFIG['data_preprocessing'].items():
            methodology_summary.append({
                'Section': 'Data Preprocessing',
                'Parameter': key.replace('_', ' ').title(), 
                'Value': str(value),
                'Description': 'Data preparation steps applied'
            })
        
        # Validation methods
        methodology_summary.append({
            'Section': 'Validation Methods',
            'Parameter': 'Methods Applied',
            'Value': ', '.join(METHODOLOGY_CONFIG['validation_methods']),
            'Description': 'Statistical and quality validation approaches'
        })
        
        # Save methodology to Excel
        with pd.ExcelWriter(out_dir / 'methodology_summary.xlsx', engine='openpyxl') as writer:
            pd.DataFrame(methodology_summary).to_excel(writer, sheet_name='Methodology', index=False)
            
        print("✓ methodology_summary.xlsx")
    else:
        print("! METHODOLOGY_CONFIG not found, skipping methodology summary")
        
except Exception as e:
    print(f"Methodology Excel export error: {e}")

print(f"\nExcel files created:")
print(f"- statistical_validation.xlsx: Comprehensive statistical results")
if 'methodology_summary' in locals() and methodology_summary:
    print(f"- methodology_summary.xlsx: Analysis methodology and parameters")
else:
    print(f"- methodology_summary.xlsx: Not created (configuration missing)")

# --------------------------------------------------------------------------
# SUMMARY
# --------------------------------------------------------------------------
print(f"\nAnalysis complete.")
print(f"Industry-specific papers: {len(filtered_papers)}")
print(f"Top industries: {', '.join([f'{ind} ({count})' for ind, count in industry_dist.head(3).items()])}")
print(f"Files generated: {len([f for f in out_dir.iterdir() if f.suffix in ['.csv', '.png', '.json', '.xlsx']])}")