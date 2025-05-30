import requests
from typing import Optional, List, Dict, Any
import json

from bujo.models.mag import MAG
import json

class Expenses:
    def __init__(self, base_url: str, api_token: str, expenses_table_id: str, mag_table_link_id: str, mag_table_instance: MAG):
        """
        Create an object of Expenses model.
        
        :param base_url: The Base URL for NocoDB App.
        :param api_token: API Token for NoCoDB.
        :param expenses_table_id: Expenses table ID in NoCoDB.
        :param mag_table_link_id: Link ID to link Expense entries to a MAG.
        """
        self.base_url = base_url.rstrip("/")
        self.table_id = expenses_table_id
        self.mag_table_link_id = mag_table_link_id
        self.mag_table_link_url = f"{self.base_url}/api/v2/tables/{self.table_id}/links/{mag_table_link_id}/records"
        self.headers = {
            "xc-token": api_token,
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        self.expenses_url = f"{self.base_url}/api/v2/tables/{self.table_id}/records"
        self.mag_table_instance = mag_table_instance

    def _url(self, path="") -> str:
        return f"{self.base_url}/api/v2/tables/{self.table_id}/records"

    def create(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        response = requests.post(self._url(), json=data, headers=self.headers)
        if response.ok:
            # Link the created expense to the MAG entry
            return response.json()
        print("❌ Create failed:", response.status_code, response.text)
        return 'failed to create expense entry. Try again?'

    def read(self, record_id: str) -> Optional[Dict[str, Any]]:
        url = f"{self._url(f'/{record_id}')}"
        response = requests.get(url, headers=self.headers)
        if response.ok:
            return response.json()
        print("❌ Read failed:", response.status_code, response.text)
        return None

    def update(self, record_id: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        response = requests.patch(self._url(f"/{record_id}"), json=data, headers=self.headers)
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

    def list(self, where: Optional[str] = None, limit: int = 1000,  sort: Optional[str] = None) -> List[Dict[str, Any]]:
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

                # Check if there are more pages
                page_info = data.get("PageInfo", {})
                if page_info.get("isLastPage", True):
                    break

                # Increment offset for the next page
                offset += limit
            else:
                print("❌ List failed:", response.status_code, response.text)
                break

        return all_results
    
    def link_mag_to_expense(self, expense_id: str, mag_id: str):
        """
        Link a MAG entry to an Expense

        :param expense_id: Expense ID to which MAG has to be Linked
        :param mag_id: MAG ID to which this expense has to be linked
        """
        payload = [{"Id": mag_id}]
        response = requests.post(f'{self.mag_table_link_url}/{expense_id}', headers=self.headers, json=payload)
        if response.ok:
            return response.text
        print("❌ List failed:", response.status_code, response.text)
        return None
