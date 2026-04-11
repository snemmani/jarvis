import logging
import requests
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class BaseNocoDB:
    def __init__(self, base_url: str, api_token: str, table_id: str):
        self.base_url = base_url.rstrip("/")
        self.table_id = table_id
        self.headers = {
            "xc-token": api_token,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _url(self, path: str = "") -> str:
        return f"{self.base_url}/api/v2/tables/{self.table_id}/records{path}"

    def create(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        response = requests.post(self._url(), json=data, headers=self.headers)
        if response.ok:
            return response.json()
        logger.error("Create failed: %s %s", response.status_code, response.text)
        return None

    def read(self, record_id: str) -> Optional[Dict[str, Any]]:
        response = requests.get(self._url(f"/{record_id}"), headers=self.headers)
        if response.ok:
            return response.json()
        logger.error("Read failed: %s %s", response.status_code, response.text)
        return None

    def delete(self, record_id: str) -> bool:
        response = requests.delete(self._url(f"/{record_id}"), headers=self.headers)
        if response.ok:
            return True
        logger.error("Delete failed: %s %s", response.status_code, response.text)
        return False

    def _paginated_list(self, params: Dict[str, Any], limit: int = 1000) -> List[Dict[str, Any]]:
        all_results: List[Dict[str, Any]] = []
        offset = 0
        while True:
            page_params = {**params, "limit": limit, "offset": offset}
            response = requests.get(self._url(), headers=self.headers, params=page_params)
            if not response.ok:
                logger.error("List failed: %s %s", response.status_code, response.text)
                break
            data = response.json()
            all_results.extend(data.get("list", []))
            if data.get("PageInfo", {}).get("isLastPage", True):
                break
            offset += limit
        return all_results
