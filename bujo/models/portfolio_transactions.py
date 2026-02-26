import requests
from typing import Optional, List, Dict, Any
import json

from bujo.models.mag import MAG


class PortfolioTransactions:
    def __init__(self, base_url: str, api_token: str, table_id: str):
        """
        Create an object of PortfolioTransactions model.
        
        :param base_url: The Base URL for NocoDB App.
        :param api_token: API Token for NoCoDB.
        :param table_id: Portfolio transactions table ID in NoCoDB.
        :param mag_table_link_id: Link ID to link transaction entries to a MAG.
        """
        self.base_url = base_url.rstrip("/")
        self.table_id = table_id
        self.headers = {
            "xc-token": api_token,
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        self.transactions_url = f"{self.base_url}/api/v2/tables/{self.table_id}/records"

    def _url(self, path: str = "", version=2) -> str:
        return f"{self.base_url}/api/v{version}/tables/{self.table_id}/records{path}"

    def create(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        response = requests.post(self._url(), json=data, headers=self.headers)
        if response.ok:
            return response.json()
        print("❌ Create failed:", response.status_code, response.text)
        return 'failed to create transaction entry. Try again?'

    def read(self, record_id: str) -> Optional[Dict[str, Any]]:
        url = self._url(f'/{record_id}')
        response = requests.get(url, headers=self.headers)
        if response.ok:
            return response.json()
        print("❌ Read failed:", response.status_code, response.text)
        return None

    def update(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        allowed_keys = {"Id", "Ticker", "TransactionType", "NoOfShares", "CostPerShare","CMP"}
        filtered_data = {key: value for key, value in data.items() if key in allowed_keys}
        response = requests.patch(self._url(), json=filtered_data, headers=self.headers)
        
        if response.ok:
            return response.json()
        print("❌ Update failed:", response.status_code, response.text)
        return None

    def delete(self, record_id: str) -> bool:
        response = requests.delete(self._url(f"/{record_id}"), headers=self.headers)
        if response.ok:
            return True
        print("❌ Delete failed:", response.status_code, response.text)
        return False

    def list(self, where: Optional[str] = None, limit: int = 1000, sort: Optional[str] = None) -> List[Dict[str, Any]]:
        all_results = []
        offset = 0
        where = json.loads(where.replace('```json', '').replace('```', ''))["filters"] if where else None
        while True:
            params = {"limit": limit, "offset": offset}
            if where:
                params["where"] = where
            if sort:
                params["sort"] = sort

            response = requests.get(self._url(), headers=self.headers, params=params)
            if response.ok:
                data = response.json()
                results = data.get("list", [])
                all_results.extend(results)

                page_info = data.get("PageInfo", {})
                if page_info.get("isLastPage", True):
                    break

                offset += limit
            else:
                print("❌ List failed:", response.status_code, response.text)
                break

        return all_results

