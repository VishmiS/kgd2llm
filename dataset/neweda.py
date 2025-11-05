import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import Counter
import re
import numpy as np


class MultiDatasetEDA:
    def __init__(self):
        self.question_patterns = {
            'What': r'^(what|What)',
            'When': r'^(when|When)',
            'Where': r'^(where|Where)',
            'Who': r'^(who|Who)',
            'Whom': r'^(whom|Whom)',
            'Which': r'^(which|Which)',
            'Whose': r'^(whose|Whose)',
            'Why': r'^(why|Why)',
            'How': r'^(how|How)(?!\s+many|\s+much|\s+old|\s+far|\s+long)',
            'How many': r'^(how many|How many)',
            'How much': r'^(how much|How much)',
            'Is/Are': r'^(is|are|Is|Are|was|were|Was|Were)',
            'Do/Does': r'^(do|does|did|Do|Does|Did)',
            'Can/Could': r'^(can|could|Can|Could)'
        }

    def classify_question_type(self, question):
        """Classify question based on starting word/pattern"""
        if pd.isna(question) or question == '':
            return 'Empty/Invalid'

        question = str(question).strip()
        for q_type, pattern in self.question_patterns.items():
            if re.search(pattern, question):
                return q_type
        return 'Other'

    def load_data(self, dataset_path):
        """Load queries from TSV file"""
        try:
            # Read with header and use the 'query' column for questions
            df = pd.read_csv(dataset_path, sep='\t', header=0)

            # Check if 'query' column exists, if not try common alternatives
            if 'query' in df.columns:
                # Rename to standardize column name
                df = df.rename(columns={'query': 'question'})
            elif 'question' in df.columns:
                # Already has question column
                pass
            elif len(df.columns) >= 2:
                # Assume second column is the question (common format: id, question, ...)
                df = df.rename(columns={df.columns[1]: 'question'})
            else:
                print(f"Could not identify question column in {dataset_path}")
                return None

            # Create question_id if not present
            if 'query_id' in df.columns:
                df = df.rename(columns={'query_id': 'question_id'})
            elif 'question_id' not in df.columns:
                df['question_id'] = range(1, len(df) + 1)

            # Keep only necessary columns and drop duplicates based on question text
            df = df[['question_id', 'question']].drop_duplicates(subset=['question']).reset_index(drop=True)

            print(f"  Loaded {len(df)} unique questions")
            return df

        except Exception as e:
            print(f"Error loading {dataset_path}: {e}")
            return None

    def analyze_dataset(self, dataset_name, dataset_path):
        """Analyze a single dataset"""
        print(f"\n{'=' * 60}")
        print(f"Analyzing {dataset_name}")
        print(f"Path: {dataset_path}")
        print(f"{'=' * 60}")

        df = self.load_data(dataset_path)
        if df is None or df.empty:
            print(f"No data found for {dataset_name}")
            return None

        # Basic statistics
        print(f"Total questions: {len(df):,}")
        print(f"Unique questions: {df['question'].nunique():,}")
        print(f"Duplicate ratio: {(1 - df['question'].nunique() / len(df)) * 100:.2f}%")

        # Check for missing values
        missing_questions = df['question'].isna().sum()
        if missing_questions > 0:
            print(f"Missing questions: {missing_questions}")

        # Classify question types
        df['question_type'] = df['question'].apply(self.classify_question_type)

        # Calculate percentages
        type_counts = df['question_type'].value_counts()
        type_percentages = (type_counts / len(df) * 100).round(2)

        # Create results dataframe
        results = pd.DataFrame({
            'Question Type': type_counts.index,
            'Count': type_counts.values,
            'Percentage': type_percentages.values
        }).sort_values('Percentage', ascending=False)

        print(f"\nTop 10 Question Types for {dataset_name}:")
        print(results.head(10).to_string(index=False))

        # Show question examples for top types
        print(f"\nExample questions for top types:")
        top_types = results.head(3)['Question Type'].tolist()
        for q_type in top_types:
            examples = df[df['question_type'] == q_type]['question'].head(2).tolist()
            print(f"  {q_type}: {examples}")

        return results, df

    def compare_datasets(self, datasets_info):
        """Compare question types across all datasets"""
        all_results = {}
        detailed_data = {}

        for dataset_name, dataset_path in datasets_info.items():
            if os.path.exists(dataset_path):
                results, df = self.analyze_dataset(dataset_name, dataset_path)
                if results is not None:
                    all_results[dataset_name] = results
                    detailed_data[dataset_name] = df
            else:
                print(f"Dataset path not found: {dataset_path}")

        if not all_results:
            print("No datasets were successfully analyzed!")
            return None, None, None

        # Create comparison table for major question types
        major_types = ['What', 'Who', 'How', 'How many', 'When', 'Where', 'Why', 'Which', 'Other']
        comparison_data = []

        for dataset_name, results in all_results.items():
            type_dict = dict(zip(results['Question Type'], results['Percentage']))
            row = {'Dataset': dataset_name, 'Total_Questions': results['Count'].sum()}

            # Add percentages for major types
            for q_type in major_types:
                row[q_type] = type_dict.get(q_type, 0)

            comparison_data.append(row)

        comparison_df = pd.DataFrame(comparison_data)

        print(f"\n{'=' * 80}")
        print("COMPARISON ACROSS ALL DATASETS")
        print(f"{'=' * 80}")
        print(comparison_df.to_string(index=False))

        return all_results, comparison_df, detailed_data

    def visualize_comparison(self, all_results, comparison_df, detailed_data=None):
        """Create visualizations comparing all datasets - Two separate graphs"""

        # Calculate total questions for each dataset
        dataset_totals = {}
        for dataset_name, results in all_results.items():
            dataset_totals[dataset_name] = results['Count'].sum()

        output_dir = "/root/pycharm_semanticsearch/plots"
        os.makedirs(output_dir, exist_ok=True)

        # 1. PIE CHART - Overall distribution (First PNG)
        plt.figure(figsize=(12, 8))

        # Get combined question data
        all_actual_questions = []
        if detailed_data:
            for dataset_name, df in detailed_data.items():
                all_actual_questions.extend(df['question_type'].tolist())
        else:
            # Fallback: use weighted approach
            for dataset_name, results in all_results.items():
                dataset_total = dataset_totals[dataset_name]
                for _, row in results.iterrows():
                    weighted_count = (row['Percentage'] / 100) * dataset_total
                    all_actual_questions.extend([row['Question Type']] * int(weighted_count))

        overall_counts = pd.Series(all_actual_questions).value_counts().head(10)

        # Clean labels by removing overlap - use autopct for percentages but customize labels
        colors = plt.cm.Set3(np.linspace(0, 1, len(overall_counts)))
        wedges, texts, autotexts = plt.pie(overall_counts.values,
                                           colors=colors,
                                           startangle=90,
                                           autopct='%1.1f%%',
                                           pctdistance=0.85,
                                           labeldistance=1.05)

        # Improve label appearance
        for text in texts:
            text.set_fontsize(10)
            text.set_fontweight('bold')

        for autotext in autotexts:
            autotext.set_fontsize(9)
            autotext.set_fontweight('bold')
            autotext.set_color('black')

        # Create legend instead of crowded labels
        plt.legend(wedges, overall_counts.index,
                   title="Question Types",
                   loc="center left",
                   bbox_to_anchor=(1, 0, 0.5, 1),
                   fontsize=10)

        # Add dataset sizes to title
        sizes_text = " | ".join([f"{name}: {total:,}" for name, total in dataset_totals.items()])
        plt.title(f'Overall Question Type Distribution\n({sizes_text})', fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'overall_distribution.png'), dpi=300, bbox_inches='tight')
        plt.show()

        # 2. HEATMAP - Dataset comparison (Second PNG)
        plt.figure(figsize=(12, 8))

        major_types = ['What', 'Who', 'How', 'How many', 'When', 'Where', 'Why', 'Which', 'Other']

        # Filter to only include types that exist in the data
        available_types = [t for t in major_types if t in comparison_df.columns]
        heatmap_data = comparison_df.set_index('Dataset')[available_types].T

        # Create heatmap
        sns.heatmap(heatmap_data,
                    annot=True,
                    fmt='.1f',
                    cmap='YlOrRd',
                    cbar_kws={'label': 'Percentage (%)'},
                    linewidths=0.5,
                    linecolor='white')

        plt.title('Question Type Percentage Heatmap\nAcross Datasets', fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Dataset', fontweight='bold')
        plt.ylabel('Question Type', fontweight='bold')
        plt.tick_params(axis='x', rotation=45)
        plt.tick_params(axis='y', rotation=0)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'dataset_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.show()

        # 3. Dataset-specific distributions (Third PNG - optional)
        self.plot_dataset_specific_distributions(all_results)

        print(f"\nVisualizations saved as separate files:")
        print(f"  - Overall distribution: {os.path.join(output_dir, 'overall_distribution.png')}")
        print(f"  - Dataset heatmap: {os.path.join(output_dir, 'dataset_heatmap.png')}")

    def plot_dataset_specific_distributions(self, all_results):
        """Create individual distribution plots for each dataset"""
        n_datasets = len(all_results)
        fig, axes = plt.subplots(1, n_datasets, figsize=(5 * n_datasets, 8))  # Increased height

        if n_datasets == 1:
            axes = [axes]

        for idx, (dataset_name, results) in enumerate(all_results.items()):
            top_10 = results.head(10)
            bars = axes[idx].barh(top_10['Question Type'], top_10['Percentage'],
                                  color=plt.cm.viridis(np.linspace(0, 1, len(top_10))))
            axes[idx].set_title(f'{dataset_name}\nTop 10 Question Types', fontweight='bold')
            axes[idx].set_xlabel('Percentage (%)', fontweight='bold')
            axes[idx].invert_yaxis()  # Highest percentage on top

            # Add value labels on bars
            for bar in bars:
                width = bar.get_width()
                axes[idx].text(width + 1, bar.get_y() + bar.get_height() / 2,
                               f'{width:.1f}%',
                               ha='left', va='center', fontsize=9)

        plt.tight_layout()
        output_dir = "/root/pycharm_semanticsearch/plots"
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'dataset_specific_distributions.png'), dpi=300, bbox_inches='tight')
        plt.show()
        print(f"  - Dataset-specific distributions: {os.path.join(output_dir, 'dataset_specific_distributions.png')}")

    def generate_comprehensive_report(self, all_results, comparison_df):
        """Generate a comprehensive text report"""
        report = []
        report.append("MULTI-DATASET QUESTION ANALYSIS - COMPREHENSIVE EDA REPORT")
        report.append("=" * 80)

        total_questions = 0
        report.append("\nDATASET SUMMARY:")
        report.append("-" * 40)

        for dataset_name, results in all_results.items():
            split_total = results['Count'].sum()
            total_questions += split_total
            top_type = results.iloc[0]['Question Type']
            top_percentage = results.iloc[0]['Percentage']

            report.append(f"\n{dataset_name.upper():<20}:")
            report.append(f"  Total questions: {split_total:,}")
            report.append(f"  Most common type: {top_type} ({top_percentage}%)")
            report.append(f"  Number of unique types: {len(results)}")

        report.append(f"\nOVERALL STATISTICS:")
        report.append("-" * 40)
        report.append(f"  Total questions across all datasets: {total_questions:,}")
        report.append(f"  Number of datasets analyzed: {len(all_results)}")

        # Cross-dataset insights
        report.append(f"\nCROSS-DATASET INSIGHTS:")
        report.append("-" * 40)

        # Find which dataset has highest percentage for each major type
        major_types = ['What', 'Who', 'How', 'How many', 'When', 'Where', 'Why']
        for q_type in major_types:
            if q_type in comparison_df.columns:
                max_idx = comparison_df[q_type].idxmax()
                max_val = comparison_df.loc[max_idx, q_type]
                dataset = comparison_df.loc[max_idx, 'Dataset']
                report.append(f"  Highest '{q_type}' questions: {dataset} ({max_val}%)")

        # Save report
        output_dir = "/root/pycharm_semanticsearch/plots"
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'multi_dataset_eda_report.txt'), 'w') as f:
            f.write('\n'.join(report))

        print("\n" + '\n'.join(report))
        print(f"\nDetailed report saved to: multi_dataset_eda_report.txt")
        print(f"Main visualization saved to: multi_dataset_eda.png")
        print(f"Dataset-specific distributions saved to: dataset_specific_distributions.png")


def main():
    # Define all dataset paths
    datasets_info = {
        'covid': '/root/pycharm_semanticsearch/dataset/covid/full/queries.tsv',
        'ms_marco': '/root/pycharm_semanticsearch/dataset/ms_marco/full30/queries.tsv',
        'web_questions': '/root/pycharm_semanticsearch/dataset/web_questions/full/queries.tsv'
    }

    # Initialize EDA analyzer
    eda = MultiDatasetEDA()

    # Perform analysis on all datasets
    all_results, comparison_df, detailed_data = eda.compare_datasets(datasets_info)

    if all_results is not None:
        # Create visualizations
        eda.visualize_comparison(all_results, comparison_df)

        # Generate comprehensive report
        eda.generate_comprehensive_report(all_results, comparison_df)

        # Create output directory
        output_dir = "/root/pycharm_semanticsearch/plots"
        os.makedirs(output_dir, exist_ok=True)

        # Save detailed results to CSV
        comparison_path = os.path.join(output_dir, 'multi_dataset_comparison.csv')
        comparison_df.to_csv(comparison_path, index=False)
        print(f"Comparison table saved to: {comparison_path}")

        # Save individual dataset results
        for dataset_name, results in all_results.items():
            dataset_path = os.path.join(output_dir, f'{dataset_name}_analysis.csv')
            results.to_csv(dataset_path, index=False)
            print(f"{dataset_name} analysis saved to: {dataset_path}")

        print(f"\nAll files saved to: {output_dir}")
    else:
        print("No datasets were successfully analyzed. Please check the file paths.")


if __name__ == "__main__":
    main()