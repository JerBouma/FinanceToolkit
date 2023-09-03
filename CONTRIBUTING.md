# Contributing
First off all, thank you for taking the time to contribute (or at least read the Contributing Guidelines)! 🚀

The goal of the Finance Toolkit is to make any type of financial calculation as transparent and efficient as possible. I want to make these type of calculations as accessible to anyone as possible and seeing how many websites exists that do the same thing (but instead you have to pay) gave me plenty of reasons to work on this.

The following is a set of guidelines for contributing to the Finance Toolkit.

## Structure

The Finance Toolkit follows the Model, View, Controller model except it cuts out the View part being more focussed on calculations than on visusally depicting the data given that there are plenty of providers that offer this solution already. The MC model (as found in `financetoolkit/base`) is therefore constructed as follows:

- **_controller** modules (such as toolkit_controller and ratios_controller) orchestrate the data flow. Through the controller, the user can set parameters (such as tickers, start and end date) that define the data that needs to be obtained. E.g. in the controller classes you will be able to find the function `get_income_statement` which collects income statements via a **_model** that takes in the parameters set by the user.
- **_model** modules (such as fundamentals_model and historical_model) are the modules that actually obtain the data. E.g in the fundamentals_model exists a function called `get_financial_statements` which would be executed by `get_income_statement` from the controller class to obtain the financial statement, in this case the income statement, for the selected parameters. These functions will also work separately, they do not need the controller to work but the controller needs them to work.

Next to that, each individual ratio, technical indicator, risk metric etc. is categorized and kept in its own directory. E.g. all functions of ratios can be found inside `financetoolkit/ratios` in which if I wanted to see the Price-to-Book ratio function, I'd visit the `valuation.py` (given that Price-to-Book ratio is categorized as a Valuation Ratio) and then scroll to the function which would be the following:

```python
def get_price_to_book_ratio(
    price_per_share: pd.Series, book_value_per_share: pd.Series
) -> pd.Series:
    """
    Calculate the price to book ratio, a valuation ratio that compares a company's market
    price to its book value per share.

    Args:
        price_per_share (float or pd.Series): Price per share of the company.
        book_value_per_share (float or pd.Series): Book value per share of the company.

    Returns:
        float | pd.Series: The price to book ratio value.
    """
    return price_per_share / book_value_per_share
```

This applies to any metric and this is also how the MC model retrieves the correct input. The `get_price_to_book_ratio` function is called by the `ratios_controller` which is called by the `toolkit_controller`. This is the flow of the data.

The separation is done so that it becomes possible to call all functions separately making the Finance Toolkit incredibly flexible for any kind of data input.

## Adding New Functionality

If you are looking to add new functionality do the following:

1. Start by looking at the available modules and whether your functionality would fit inside one of the modules.
    - If the answer is yes, add it to this module.
    - If the answer is no, create a new module and add it there.
2. Figure out whether your functionality would fit with an existing controller. E.g. did you add a new ratio? Consider adding it to the `ratios_controller`.
    - If the answer is yes, add it to this controller.
    - If the answer is no, create a new controller and add it there. Make sure to also connect the controller to the `toolkit_controller` so that it can be called from there.
3. Add in the relevant docstrings (be as extensive as possible, following already created examples) and update the README if relevant.
4. Add in the relevant tests (see `tests` directory for examples).
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