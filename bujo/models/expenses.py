import json
import logging
import requests
from typing import Any, Dict, List, Optional

from bujo.models.base import BaseNocoDB
from bujo.models.mag import MAG

logger = logging.getLogger(__name__)


class Expenses(BaseNocoDB):
    def __init__(
        self,
        base_url: str,
        api_token: str,
        expenses_table_id: str,
        mag_table_link_id: str,
        mag_table_instance: MAG,
    ):
        super().__init__(base_url, api_token, expenses_table_id)
        self.mag_table_link_id = mag_table_link_id
        self.mag_table_link_url = (
            f"{self.base_url}/api/v2/tables/{self.table_id}/links/{mag_table_link_id}/records"
        )
        self.mag_table_instance = mag_table_instance

    def create(self, data: Dict[str, Any]) -> Any:
        response = requests.post(self._url(), json=data, headers=self.headers)
        if response.ok:
            return response.json()
        logger.error("Create failed: %s %s", response.status_code, response.text)
        return "failed to create expense entry. Try again?"

    def update(self, record_id: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        response = requests.patch(self._url(f"/{record_id}"), json=data, headers=self.headers)
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

    def link_mag_to_expense(self, expense_id: str, mag_id: str) -> Optional[str]:
        payload = [{"Id": mag_id}]
        response = requests.post(
            f"{self.mag_table_link_url}/{expense_id}", headers=self.headers, json=payload
        )
        if response.ok:
            return response.text
        logger.error("Link MAG to expense failed: %s %s", response.status_code, response.text)
        return None
