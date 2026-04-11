import json
import logging
import requests
from typing import Any, Dict, List, Optional

from bujo.models.base import BaseNocoDB

logger = logging.getLogger(__name__)

# Fields the LLM is allowed to modify via Update MAG.
_UPDATABLE_FIELDS = {"Note", "Exercise"}


class MAG(BaseNocoDB):
    def __init__(self, base_url: str, api_token: str, mag_table_id: str):
        super().__init__(base_url, api_token, mag_table_id)

    def update(self, data: str) -> str:
        parsed = json.loads(data.replace("```json", "").replace("```", ""))
        mag_object = self.find_by_date(parsed["date_filter"])
        if not mag_object:
            return "Failed to find MAG object with the given date filter."
        mag_object.update({k: v for k, v in parsed["payload"].items() if k in _UPDATABLE_FIELDS})
        payload = {"Id": mag_object["Id"]}
        payload.update({k: mag_object[k] for k in _UPDATABLE_FIELDS if k in mag_object})
        response = requests.patch(self._url(), json=payload, headers=self.headers)
        if response.ok:
            return response.text
        logger.error("Update failed: %s %s", response.status_code, response.text)
        return "Updating MAG failed. Try again?"

    def list(self, where: Optional[str] = None, sort: Optional[str] = None) -> List[Dict[str, Any]]:
        parsed = json.loads(where.replace("```json", "").replace("```", "")) if where else {}
        filters = parsed.get("filters")
        params: Dict[str, Any] = {}
        if filters:
            params["where"] = filters
        if sort:
            params["sort"] = sort
        return self._paginated_list(params)

    def find_by_date(self, iso_date_str: str) -> Optional[Dict[str, Any]]:
        params = {"where": f"(Date,eq,exactDate,{iso_date_str})"}
        response = requests.get(self._url(), headers=self.headers, params=params)
        if response.ok:
            items = response.json().get("list", [])
            return items[0] if items else None
        logger.error("Search by date failed: %s %s", response.status_code, response.text)
        return None
