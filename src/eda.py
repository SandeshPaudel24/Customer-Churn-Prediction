import os
import warnings

import matplotlib.pyplot as plt
import missingno as msno
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

plt.style.use("ggplot")

class EDA:

    def __init__(self, dataframe):

        self.df = dataframe

        self.output = "outputs/eda"

        os.makedirs(self.output, exist_ok=True)

    ##################################################

    def dataset_summary(self):

        print("\n" + "="*60)
        print("DATASET SUMMARY")
        print("="*60)

        print(self.df.describe(include="all"))

        summary = self.df.describe(include="all")

        summary.to_csv(
            os.path.join(
                self.output,
                "dataset_summary.csv"
            )
        )

    ##################################################

    def class_distribution(self):

        plt.figure(figsize=(6,5))

        sns.countplot(
            data=self.df,
            x="Churn"
        )

        plt.title("Customer Churn Distribution")

        plt.tight_layout()

        plt.savefig(

            os.path.join(

                self.output,

                "churn_distribution.png"

            )

        )

        plt.close()

    ##################################################

    def missing_values(self):

        plt.figure(figsize=(12,6))

        msno.matrix(self.df)

        plt.savefig(

            os.path.join(

                self.output,

                "missing_values.png"

            )

        )

        plt.close()

    ##################################################

    def correlation_heatmap(self):

        plt.figure(figsize=(18,12))

        corr = self.df.corr(numeric_only=True)

        sns.heatmap(

            corr,

            cmap="coolwarm",

            center=0

        )

        plt.title("Correlation Heatmap")

        plt.tight_layout()

        plt.savefig(

            os.path.join(

                self.output,

                "correlation_heatmap.png"

            )

        )

        plt.close()
        
    ##################################################

    def numerical_histograms(self):

        numerical = self.df.select_dtypes(

            include=["int64","float64"]

        )

        numerical.hist(

            figsize=(18,12),

            bins=20

        )

        plt.tight_layout()

        plt.savefig(

            os.path.join(

                self.output,

                "histograms.png"

            )

        )

        plt.close()

    ##################################################

    def boxplots(self):

        numerical = self.df.select_dtypes(

            include=["int64","float64"]

        )

        for column in numerical.columns:

            plt.figure(figsize=(6,4))

            sns.boxplot(

                y=self.df[column]

            )

            plt.title(column)

            plt.tight_layout()

            plt.savefig(

                os.path.join(

                    self.output,

                    f"{column}_boxplot.png"

                )

            )

            plt.close()

    ##################################################

    def tenure_vs_churn(self):

        plt.figure(figsize=(8,5))

        sns.boxplot(

            data=self.df,

            x="Churn",

            y="tenure"

        )

        plt.tight_layout()

        plt.savefig(

            os.path.join(

                self.output,

                "tenure_vs_churn.png"

            )

        )

        plt.close()

    ##################################################

    def monthlycharge_vs_churn(self):

        plt.figure(figsize=(8,5))

        sns.boxplot(

            data=self.df,

            x="Churn",

            y="MonthlyCharges"

        )

        plt.tight_layout()

        plt.savefig(

            os.path.join(

                self.output,

                "monthlycharges_vs_churn.png"

            )

        )

        plt.close()
        
    ##################################################

    def contract_distribution(self):

        contract_cols = [

            c for c in self.df.columns

            if "Contract_" in c

        ]

        if len(contract_cols) == 0:

            return

        contract = self.df[contract_cols].sum()

        plt.figure(figsize=(8,5))

        contract.plot(kind="bar")

        plt.title("Contract Types")

        plt.tight_layout()

        plt.savefig(

            os.path.join(

                self.output,

                "contract_types.png"

            )

        )

        plt.close()

    ##################################################

    def internet_distribution(self):

        internet_cols = [

            c for c in self.df.columns

            if "InternetService_" in c

        ]

        if len(internet_cols)==0:

            return

        internet = self.df[internet_cols].sum()

        plt.figure(figsize=(8,5))

        internet.plot(kind="bar")

        plt.title("Internet Services")

        plt.tight_layout()

        plt.savefig(

            os.path.join(

                self.output,

                "internet_services.png"

            )

        )

        plt.close()
        
    ##################################################

    def run(self):

        print("\nGenerating EDA...\n")

        self.dataset_summary()

        self.class_distribution()

        self.missing_values()

        self.correlation_heatmap()

        self.numerical_histograms()

        self.boxplots()

        self.tenure_vs_churn()

        self.monthlycharge_vs_churn()

        self.contract_distribution()

        self.internet_distribution()

        print("\nEDA Completed Successfully")