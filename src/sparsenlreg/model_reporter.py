from pathlib import Path


class ModelReporter:
    """Handles reporting and displaying model summaries in Markdown format."""

    def __init__(self):
        pass

    def generate_markdown_report(self, best_models, X_names, y_name, export_dir: Path):
        """
        Generate, print, and export a Markdown summary of the best models.

        Args:
            best_models: List of the best models.
            X_names: List of feature names.
            y_name: Name of the output variable.
            export_dir (Path): Directory to store the report.
        """
        # Generate the header
        header = "# SpareNLReg report\n\n"
        # Generate the markdown summary table
        markdown_summary_table = self._generate_markdown_summary_table(
            best_models, X_names, y_name
        )
        # Generate the markdown for best models
        best_models_summary = self._generate_best_models_summary(
            best_models, X_names, y_name
        )
        # Generate the latex summary table
        latex_summary_table = self._generate_latex_summary_table(
            best_models, X_names, y_name
        )

        # Export the Markdown summary to a file
        with open(export_dir / "report.md", "w") as f:
            f.write(header)
            f.write(markdown_summary_table)
            f.write(latex_summary_table)
            f.write(best_models_summary)

    def _generate_best_models_summary(self, best_models, X_names, y_name) -> str:
        """
        Generate a Markdown-formatted summary for the best models.

        Args:
            best_models: List of the best models.
            X_names: List of feature names.
            y_name: Name of the output variable.

        Returns:
            str: Markdown-formatted summary string.
        """
        markdown_lines = []
        markdown_lines.append("## Best Models\n")

        for model_data in best_models:
            cst = model_data.intercept_
            coeffs = model_data.coef_
            N_coefs = model_data.N_coefs
            r2 = model_data.r2_score
            beta = model_data.alpha

            # Model header
            markdown_lines.append(f"### Model {N_coefs}")

            # Display the LaTeX expression as a rendered equation
            markdown_lines.append("**Expression:**")
            latex_expr = f"${y_name} = {cst:+.3g}"
            for i, coeff in enumerate(coeffs):
                if not coeff == 0:
                    latex_expr += f" {coeff:+.3g} {X_names[i]}"
            latex_expr += "$"
            markdown_lines.append(latex_expr)

            # Assessment
            markdown_lines.append("\n**Assessment:**\n")
            markdown_lines.append(f"- Hyperparameter $\\beta = {beta:1.2e}$")
            markdown_lines.append(f"- Number of coefficients: `{N_coefs}`")
            markdown_lines.append(f"- Score: $R^2 = {r2:.3f}$")
            markdown_lines.append("")

            # LaTeX expression (raw code block)
            markdown_lines.append("\n**LaTeX Code:**")
            markdown_lines.append(f"```latex\n{y_name} = {cst:+.3g}")
            for i, coeff in enumerate(coeffs):
                if not coeff == 0:
                    markdown_lines.append(f"    {coeff:+.3g} {X_names[i]}")
            markdown_lines.append("```")

            # Python function (raw code block)
            markdown_lines.append("\n**Python function:**")
            markdown_lines.append("```python")
            markdown_lines.append(f"\ndef {y_name}(TODO):")
            markdown_lines.append("    return (")
            markdown_lines.append(f"        {cst:+.3g}")
            for i, coeff in enumerate(coeffs):
                if not coeff == 0:
                    X_name = X_names[i].replace(" ", "*")
                    markdown_lines.append(
                        f"        {coeff}*{X_name}".replace("^", "**")
                    )
            markdown_lines.append("    )")
            markdown_lines.append("```")

            # Separator
            markdown_lines.append("")

        return "\n".join(markdown_lines)

    def _generate_latex_summary_table(self, best_models, X_names, y_name):
        # Initialize the content
        markdown_lines = []
        # Add an header
        markdown_lines.append("## LaTeX summary table\n")
        markdown_lines.append("```latex")

        latex_table = []
        for model_data in best_models:
            cst = model_data.intercept_
            coeffs = model_data.coef_
            N_coefs = model_data.N_coefs
            r2 = model_data.r2_score
            beta = model_data.alpha

            # Build the LaTeX expression for the model
            latex_expr = f"{y_name}(a) = {cst:+.3g}"
            for i, coeff in enumerate(coeffs):
                if not coeff == 0:
                    latex_expr += f" {coeff:+.3g} {X_names[i]}"

            # Append the row to the table
            latex_table.append(
                f"{r2:.3f} & {N_coefs} & {beta:.2g} & ${latex_expr}$ \\\\"
            )

        # Generate the LaTeX table
        markdown_lines.append(
            "\\begin{tabular}{|c|c|c|l|}\n"
            "\\hline\n"
            "$R^2$ & $N_{\\mathrm{coeff}}$ & $\\beta$ & Model \\\\\n"
            "\\hline"
        )

        markdown_lines.append("\n".join(latex_table))

        markdown_lines.append("\\\\\n\\hline\n\\end{tabular}\n")
        markdown_lines.append("```")
        markdown_lines.append("")

        return "\n".join(markdown_lines)

    def _generate_markdown_summary_table(self, best_models, X_names, y_name) -> str:
        """
        Generate a Markdown-formatted table summarizing the best models.

        Args:
            best_models: List of dictionaries containing the best models.
            X_names: List of feature names.
            y_name: Name of the output variable.

        Returns:
            str: Markdown-formatted table.
        """
        markdown_lines = []
        markdown_lines.append("## Markdown Summary Table\n")  # Title

        # Table header
        markdown_lines.append("| $R^2$ | $N_{\\mathrm{coeff}}$ | $\\beta$ | Model |")
        markdown_lines.append("|--|--|--|--------------|")

        # Table rows
        for model_data in best_models:
            cst = model_data.intercept_
            coeffs = model_data.coef_
            N_coefs = model_data.N_coefs
            r2 = model_data.r2_score
            beta = model_data.alpha

            # Build the model expression
            model_expr = f"{y_name} = {cst:+.3g}"
            for i, coeff in enumerate(coeffs):
                if not coeff == 0:
                    model_expr += f" {coeff:+.3g} {X_names[i]}"

            # Append the row to the table
            markdown_lines.append(
                f"| {r2:<5.3f} | {N_coefs:<2} | {beta:<7.2g} | ${model_expr}$ |"
            )
        markdown_lines.append("")

        return "\n".join(markdown_lines)

    def display_terminal_table(self, best_models, X_names, y_name) -> str:
        """
        Display a concise table for terminal printing with fixed column widths.

        Args:
            best_models: List of dictionaries containing the best models.
            X_names: List of feature names.
            y_name: Name of the output variable.
        """
        # Define fixed column widths
        r2_width = 6
        n_coeff_width = 7
        beta_width = 7
        model_width = 79 - r2_width - n_coeff_width - beta_width

        # Create the table header
        header = f"{'R²':^{r2_width}} | {'N_coefs':^{n_coeff_width}} | {'β':^{beta_width}} | {'Model':<{model_width}}"
        separator = f"{'-' * r2_width}-|-{'-' * n_coeff_width}-|-{'-' * beta_width}-|-{'-' * model_width}"

        # Create the table rows
        rows = []
        for model_data in best_models:
            cst = model_data.intercept_
            coeffs = model_data.coef_
            N_coefs = model_data.N_coefs
            r2 = model_data.r2_score
            beta = model_data.alpha

            # Format the first line with R², N_coefs, and β
            first_line = f"{r2:>{r2_width}.3f} | {N_coefs:^{n_coeff_width}} | {beta:^{beta_width}.2g} | {y_name} = {cst:+.3g}"
            rows.append(first_line)

            # Add each term of the model expression on its own line
            for i, coeff in enumerate(coeffs):
                if not coeff == 0:
                    # Empty columns for R², N_coefs, and β
                    term_line = f"{'':<{r2_width}} | {'':<{n_coeff_width}} | {'':<{beta_width}} |   {coeff:+.3g} {X_names[i]}"
                    rows.append(term_line)

            # Add a row separator after each model
            rows.append(separator)

        # Remove the last separator
        if rows and rows[-1] == separator:
            rows.pop()

        # Combine all parts of the table
        table_lines = []
        table_lines.append(header)
        table_lines.append(separator)
        table_lines.extend(rows)

        print("\n".join(table_lines))
