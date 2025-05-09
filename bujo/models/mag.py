import requests
from typing import Optional, List, Dict, Any
import json

from bujo import mag


class MAG:
    def __init__(self, base_url: str, api_token: str, mag_table_id: str):
        """
        Initializes the instance with the provided base URL, API token, and table ID.
        Args:
            base_url (str): The base URL for the API endpoint. Trailing slashes will be removed.
            api_token (str): The API token used for authentication.
            mag_table_id (str): The ID of the table to interact with.
        Attributes:
            base_url (str): The sanitized base URL without trailing slashes.
            table_id (str): The ID of the table to interact with, initialized as an empty string.
            headers (dict): A dictionary containing the headers for API requests, including the
                            API token, content type, and accepted response format.
        """

        self.base_url = base_url.rstrip("/")
        self.table_id = mag_table_id
        self.headers = {
            "xc-token": api_token,
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

    def _url(self, path="") -> str:
        return f"{self.base_url}/api/v2/tables/{self.table_id}/records{path}"

    def create(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        response = requests.post(self._url(), json=data, headers=self.headers)
        if response.ok:
            return response.json()
        print("❌ Create failed:", response.status_code, response.text)
        return None

    def read(self, record_id: str) -> Optional[Dict[str, Any]]:
        url = self._url(f"/{record_id}")
        response = requests.get(url, headers=self.headers)
        if response.ok:
            return response.json()
        print("❌ Read failed:", response.status_code, response.text)
        return None

    def update(self, data: str) -> Optional[Dict[str, Any]]:
        data = json.loads(data.replace('```json', '').replace('```', ''))
        mag_object = self.find_by_date(data['date_filter'])
        if not mag_object:
            return "Failed to find MAG object with the given date filter."
        allowed_keys = {"Id", "Date", "Note", "Tithi", "Exercise"}
        mag_object.update(data['payload'])
        filtered_data = {key: value for key, value in mag_object.items() if key in allowed_keys}
        response = requests.patch(self._url(), json=filtered_data, headers=self.headers)
        if response.ok:
            return response.text
        print("❌ Update failed:", response.status_code, response.text)
        return 'Updating MAG failed. Try again?'

    def delete(self, record_id: str) -> bool:
        response = requests.delete(self._url(f"/{record_id}"), headers=self.headers)
        if response.ok:
            return True
        print("❌ Delete failed:", response.status_code, response.text)
        return False

    def list(self, where: Optional[str] = None, limit: int = 25,  sort: Optional[str] = None) -> List[Dict[str, Any]]:
        where = json.loads(where.replace('```json', '').replace('```', ''))['filters'] if where else None
        params = {"limit": limit}
        if where:
            params["where"] = where
        if sort:
            params["sort"] = sort
        response = requests.get(self._url(), headers=self.headers, params=params)
        if response.ok:
            return response.json().get("list", [])
        print("❌ List failed:", response.status_code, response.text)
        return []

    def find_by_date(self, iso_date_str: str) -> Optional[Dict[str, Any]]:
        """
        Search MAG record by ISO-formatted date string (e.g., "2025-04-10")
        """
        params = {"where": f"(Date,eq,exactDate,{iso_date_str})"}
        response = requests.get(self._url(), headers=self.headers, params=params)
        if response.ok:
            items = response.json().get("list", [])
            return items[0] if items else None
        print("❌ Search by date failed:", response.status_code, response.text)
        return None
