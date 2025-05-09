import unittest
from unittest.mock import patch, MagicMock
from bujo.models.mag import MAG


class TestMAG(unittest.TestCase):
    def setUp(self):
        self.base_url = "https://example.com"
        self.api_token = "test_token"
        self.mag_table_id = "test_table_id"
        self.mag = MAG(self.base_url, self.api_token, self.mag_table_id)

    @patch("requests.post")
    def test_create_success(self, mock_post):
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {"id": "123", "name": "Test"}
        mock_post.return_value = mock_response

        data = {"name": "Test"}
        result = self.mag.create(data)

        self.assertEqual(result, {"id": "123", "name": "Test"})
        mock_post.assert_called_once_with(
            f"{self.base_url}/api/v2/tables/{self.mag_table_id}/records",
            json=data,
            headers=self.mag.headers,
        )

    @patch("requests.get")
    def test_read_success(self, mock_get):
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {"id": "123", "name": "Test"}
        mock_get.return_value = mock_response

        record_id = "123"
        result = self.mag.read(record_id)

        self.assertEqual(result, {"id": "123", "name": "Test"})
        mock_get.assert_called_once_with(
            f"{self.base_url}/api/v2/tables/{self.mag_table_id}/records/{record_id}",
            headers=self.mag.headers,
        )

    @patch("requests.patch")
    def test_update_success(self, mock_patch):
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {"id": "123", "name": "Updated"}
        mock_patch.return_value = mock_response

        record_id = "123"
        data = {"name": "Updated"}
        result = self.mag.update(record_id, data)

        self.assertEqual(result, {"id": "123", "name": "Updated"})
        mock_patch.assert_called_once_with(
            f"{self.base_url}/api/v2/tables/{self.mag_table_id}/records/{record_id}",
            json=data,
            headers=self.mag.headers,
        )

    @patch("requests.delete")
    def test_delete_success(self, mock_delete):
        mock_response = MagicMock()
        mock_response.ok = True
        mock_delete.return_value = mock_response

        record_id = "123"
        result = self.mag.delete(record_id)

        self.assertTrue(result)
        mock_delete.assert_called_once_with(
            f"{self.base_url}/api/v2/tables/{self.mag_table_id}/records/{record_id}",
            headers=self.mag.headers,
        )

    @patch("requests.get")
    def test_list_success(self, mock_get):
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {"list": [{"id": "123", "name": "Test"}]}
        mock_get.return_value = mock_response

        result = self.mag.list(limit=10)

        self.assertEqual(result, [{"id": "123", "name": "Test"}])
        mock_get.assert_called_once_with(
            f"{self.base_url}/api/v2/tables/{self.mag_table_id}/records",
            headers=self.mag.headers,
            params={"limit": 10},
        )

    @patch("requests.get")
    def test_find_by_date_success(self, mock_get):
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {"list": [{"id": "123", "date": "2025-04-10"}]}
        mock_get.return_value = mock_response

        iso_date_str = "2025-04-10"
        result = self.mag.find_by_date(iso_date_str)

        self.assertEqual(result, {"id": "123", "date": "2025-04-10"})
        mock_get.assert_called_once_with(
            f"{self.base_url}/api/v2/tables/{self.mag_table_id}/records",
            headers=self.mag.headers,
            params={"where": f"(Date,eq,exactDate,{iso_date_str})"},
        )


if __name__ == "__main__":
    unittest.main()
