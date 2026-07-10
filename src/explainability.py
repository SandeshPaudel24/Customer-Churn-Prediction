import os
import shap
import joblib
import matplotlib.pyplot as plt


class Explainability:

    def __init__(self, model, X_train, X_test):

        self.model = model
        self.X_train = X_train
        self.X_test = X_test

        os.makedirs("outputs/shap", exist_ok=True)

    ###########################################################

    def create_explainer(self):
        print("\nCreating SHAP Explainer...")

        if self.model.__class__.__name__ in ["RandomForestClassifier", "RandomForestRegressor",
                                            "XGBClassifier", "XGBRegressor",
                                            "DecisionTreeClassifier", "DecisionTreeRegressor"]:
            self.explainer = shap.TreeExplainer(self.model)
        else:
            self.explainer = shap.Explainer(self.model, self.X_test)

        self.shap_values = self.explainer(self.X_test) if not hasattr(self.explainer, "shap_values") else self.explainer.shap_values(self.X_test)

        print("Explainer Created.")

    ###########################################################

    def summary_plot(self):

        print("Generating SHAP Summary Plot...")

        plt.figure()

        shap.summary_plot(

            self.shap_values,

            self.X_test,

            show=False

        )

        plt.tight_layout()

        plt.savefig(

            "outputs/shap/shap_summary.png",

            dpi=300,

            bbox_inches="tight"

        )

        plt.close()

    ###########################################################

    def bar_plot(self):

        print("Generating Feature Importance...")

        plt.figure()

        shap.summary_plot(

            self.shap_values,

            self.X_test,

            plot_type="bar",

            show=False

        )

        plt.tight_layout()

        plt.savefig(

            "outputs/shap/shap_feature_importance.png",

            dpi=300,

            bbox_inches="tight"

        )

        plt.close()

    ###########################################################

    def waterfall_plot(self):

        print("Generating Waterfall Plot...")

        explanation = shap.Explanation(

            values=self.shap_values[0],

            base_values=self.explainer.expected_value,

            data=self.X_test.iloc[0],

            feature_names=self.X_test.columns

        )

        plt.figure()

        shap.plots.waterfall(

            explanation,

            show=False

        )

        plt.savefig(

            "outputs/shap/shap_waterfall.png",

            dpi=300,

            bbox_inches="tight"

        )

        plt.close()

    ###########################################################

    def dependence_plot(self):

        print("Generating Dependence Plot...")

        feature = self.X_test.columns[0]

        shap.dependence_plot(

            feature,

            self.shap_values,

            self.X_test,

            show=False

        )

        plt.savefig(

            "outputs/shap/shap_dependence.png",

            dpi=300,

            bbox_inches="tight"

        )

        plt.close()

    ###########################################################

    def run(self):

        self.create_explainer()

        self.summary_plot()

        self.bar_plot()

        self.waterfall_plot()

        self.dependence_plot()

        print("\nSHAP Analysis Completed.")