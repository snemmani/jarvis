from bujo.base import expenses_model, mag_model, portfolio_transactions_model
from bujo.expenses.manage import ExpenseManager
from bujo.mag.manage import MagManager
from bujo.portoflio.manage import PortfolioManager

expense_manager   = ExpenseManager(expenses_model, mag_model)
mag_manager       = MagManager(mag_model)
portfolio_manager = PortfolioManager(portfolio_transactions_model)
