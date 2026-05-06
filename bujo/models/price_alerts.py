import logging
import requests
from typing import Any, Dict, List, Optional

from bujo.models.base import BaseNocoDB

logger = logging.getLogger(__name__)


class PriceAlerts(BaseNocoDB):
    def create(self, ticker: str, direction: str, target_price: float, action: str = "") -> Optional[Dict[str, Any]]:
        data = {"Ticker": ticker, "Direction": direction, "TargetPrice": target_price, "Active": True}
        if action:
            data["Action"] = action
        response = requests.post(self._url(), json=data, headers=self.headers)
        if response.ok:
            return response.json()
        logger.error("PriceAlerts create failed: %s %s", response.status_code, response.text)
        return None

    def list_active(self) -> List[Dict[str, Any]]:
        rows = self._paginated_list({})
        return [r for r in rows if r.get("Active")]

    def update(self, alert_id: int, **fields) -> bool:
        response = requests.patch(
            self._url(f"/{alert_id}"),
            json=fields,
            headers=self.headers,
        )
        if not response.ok:
            logger.error("PriceAlerts update failed: %s %s", response.status_code, response.text)
        return response.ok

    def deactivate(self, alert_id: int) -> bool:
        response = requests.patch(
            self._url(f"/{alert_id}"),
            json={"Active": False},
            headers=self.headers,
        )
        if not response.ok:
            logger.error("PriceAlerts deactivate failed: %s %s", response.status_code, response.text)
        return response.ok
