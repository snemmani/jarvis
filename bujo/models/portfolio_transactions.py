import json
import logging
import requests
from typing import Any, Dict, List, Optional

from bujo.models.base import BaseNocoDB

logger = logging.getLogger(__name__)

_ALLOWED_UPDATE_KEYS = {"Id", "Ticker", "TransactionType", "NoOfShares", "CostPerShare", "CMP"}


class PortfolioTransactions(BaseNocoDB):
    def __init__(self, base_url: str, api_token: str, table_id: str):
        super().__init__(base_url, api_token, table_id)

    def create(self, data: Dict[str, Any]) -> Any:
        response = requests.post(self._url(), json=data, headers=self.headers)
        if response.ok:
            return response.json()
        logger.error("Create failed: %s %s", response.status_code, response.text)
        return "failed to create transaction entry. Try again?"

    def update(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        filtered = {k: v for k, v in data.items() if k in _ALLOWED_UPDATE_KEYS}
        response = requests.patch(self._url(), json=filtered, headers=self.headers)
        if response.ok:
            return response.json()
        logger.error("Update failed: %s %s", response.status_code, response.text)
        return None

    def list(self, where: Optional[str] = None, limit: int = 1000, sort: Optional[str] = None) -> List[Dict[str, Any]]:
        parsed = json.loads(where.replace("```json", "").replace("```", "")) if where else {}
        filters = parsed.get("filters")
        params: Dict[str, Any] = {}
        if filters:
            params["where"] = filters
        if sort:
            params["sort"] = sort
        return self._paginated_list(params, limit)
