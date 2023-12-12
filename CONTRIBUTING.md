# Contributing
First off all, thank you for taking the time to contribute (or at least read the Contributing Guidelines)! 🚀

The goal of the Finance Toolkit is to make any type of financial calculation as transparent and efficient as possible. I want to make these type of calculations as accessible to anyone as possible and seeing how many websites exists that do the same thing (but instead you have to pay) gave me plenty of reasons to work on this.
___ 

<b><div align="center">Looking to learn how to create and maintain financial models with Python? Have a look at my in-depth guide <a href="https://www.jeroenbouma.com//modelling/introduction">here</a>.</div></b>
___

## Structure

The Finance Toolkit follows the Model, View and Controller (MVC) pattern. This is a pattern in software design commonly used to implement user interfaces, data, and controlling logic. It emphasizes a separation between the software’s business logic and display. This “separation of concerns” provides for a better division of labor and improved maintenance. The Finance Toolkit utilizes only the Controller and Model logic.

- **_controller** modules (such as toolkit_controller and ratios_controller) orchestrate the data flow. Through the controller, the user can set parameters (such as tickers, start and end date) that define the data that needs to be obtained. E.g. in the controller classes you will be able to find the function `get_income_statement` which collects income statements via a **_model** that takes in the parameters set by the user.
- **_model** modules (such as fundamentals_model and historical_model) are the modules that actually obtain the data. E.g in the fundamentals_model exists a function called `get_financial_statements` which would be executed by `get_income_statement` from the controller class to obtain the financial statement, in this case the income statement, for the selected parameters. These functions will also work separately, they do not need the controller to work but the controller needs them to work.

Each model function is categorized in a specific module. For example, the Gross Margin calculation is categorized under the `profitability_model.py` module which contains all of the other profitability ratios. The same applies to the other ratio categories such as liquidity, solvency, efficiency and valuation which can be found in `liquidity_model.py`, `solvency_model.py`, `efficiency_model.py` and `valuation_model.py` respectively.

```python
def get_gross_margin(revenue: pd.Series, cost_of_goods_sold: pd.Series) -> pd.Series:
    """
    Calculate the gross margin, a profitability ratio that measures the percentage of
    revenue that exceeds the cost of goods sold.

    Args:
        revenue (float or pd.Series): Total revenue of the company.
        cost_of_goods_sold (float or pd.Series): Total cost of goods sold of the company.

    Returns:
        float | pd.Series: The gross margin percentage value.
    """
    return (revenue - cost_of_goods_sold) / revenue
```

As seen in my [Financial Modeling with Python Guide](https://www.jeroenbouma.com/modelling/structure-your-model), the Model, View and Controller for Gross Margin calculation will be named `profitability_model.py`, `profitability_view.py` and `profitability_controller.py` respectively. The `helpers.py` module will be placed in the root of the package.

The separation is done so that it becomes possible to call all functions separately making the Finance Toolkit incredibly flexible for any kind of data input. See how this would look like the following example:

```mermaid
flowchart TB;
classDef boxfont fill:#3b9cba,stroke-width:0px,fill-opacity:0.7,color:white,radius:20px;

Step0["User"] -- <b>Step 1<br></b>Initializes the FinanceToolkit -->  Step1["Toolkit Controller"]:::boxfont
Step1["Toolkit Controller"] <-- <b>Step 2<br></b>Asks for Fundamental Data --> Step2a["Fundamentals Model"]:::boxfont
Step1["Toolkit Controller"] <-- <b>Step 3<br></b>Asks for Historical Data --> Step2b["Historical Model"]:::boxfont
Step1["Toolkit Controller"] -- <b>Step 4<br></b>Initializes the Ratios Controller --> Step3["Ratios Controller"]:::boxfont
Step3["Ratios Controller"] -- <b>Step 5a<br></b>Calculates the Gross Margin --> Step2["Profitability Model"]:::boxfont
Step3["Ratios Controller"] -- <b>Step 5b<br></b>Optional Growth Calculation --> Step4["Helpers"]:::boxfont
Step3["Ratios Controller"] -- <b>Step 6<br></b>Shows the Gross Margin Data --> Step0["User"]:::boxfont
```

## Adding New Functionality

If you are looking to add new functionality do the following:

1. Start by looking at the available modules and whether your functionality would fit inside one of the modules.
    - If the answer is yes, add it to this module.
    - If the answer is no, create a new module and add it there.
2. Figure out whether your functionality would fit with an existing controller. E.g. did you add a new ratio? Consider adding it to the `ratios_controller`.
    - If the answer is yes, add it to this controller.
    - If the answer is no, create a new controller and add it there. Make sure to also connect the controller to the `toolkit_controller` so that it can be called from there.
3. Add in the relevant docstrings (be as extensive as possible, following already created examples) and update the README if relevant.
4. Add in the relevant tests (see `tests` directory for examples). If this is too difficult, feel free to skip.
5. Create a Pull Request with your new additions. See the next section how to do so.

## Working with Git & Pull Requests

Any new contribution preferably goes via a Pull Request. In essence, all you really need is Git and basic understanding of how a Pull Request works. Find some resources that explain this well here:

- [Finding ways to contribute to open source on GitHub](https://docs.github.com/en/get-started/exploring-projects-on-github/finding-ways-to-contribute-to-open-source-on-github)
- [Set up Git](https://docs.github.com/en/get-started/quickstart/set-up-git)
- [Collaborating with pull requests](https://docs.github.com/en/github/collaborating-with-pull-requests)

On every Pull Request, a couple of linters will run (see [here](https://github.com/JerBouma/FinanceToolkit/tree/main/.github/workflows) as well as unit tests for each function in the package. The linters check the code and whether it matches specific coding formatting. The tests check whether running the function returns the expected output. If any of these fail, the Pull Request can not be merged.

## Following the Workflow

After setting up Git, you can fork and pull the project in.

1. Fork the Project ([more info](https://docs.github.com/en/get-started/quickstart/fork-a-repo))
    - **Using GitHub Desktop:** [Getting started with GitHub Desktop](https://docs.github.com/en/desktop/installing-and-configuring-github-desktop/getting-started-with-github-desktop) will guide you through setting up Desktop. Once Desktop is set up, you can use it to [fork the repo](https://docs.github.com/en/desktop/contributing-and-collaborating-using-github-desktop/cloning-and-forking-repositories-from-github-desktop)!
    - **Using the command line:** [Fork the repo](https://docs.github.com/en/github/getting-started-with-github/fork-a-repo#fork-an-example-repository) so that you can make your changes without affecting the original project until you're ready to merge them.
2. Pull the Repository Locally ([more info](https://github.com/git-guides/git-pull))
3. Create your own branch (`git checkout -b feature/contribution`)
4. Add your changes (`git add .`)
5. Install pre-commit, this checks the code for any errors before committing (`pre-commit install`)
6. Commit your Changes (`git commit -m 'Improve the Toolkit'`)
7. Check whether the tests still pass (`pytest tests`) and if not, correct then.
    - When no formulas have changed or new tests have been added, you can use `pytest tests --record-mode=rewrite` (please do provide reasoning in this case).
    - If formulas or calculations have changed, adjusts the tests inside the `tests` directory.
8. Push to your Branch (`git push origin feature/contribution`)
9. Open a Pull Request

**Note:** feel free to reach out if you run into any issues: jer.bouma@gmail.com or [LinkedIn](https://www.linkedin.com/in/boumajeroen/) or open a GitHub Issue.
