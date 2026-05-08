import logging
import requests
from typing import Any, Dict, List, Optional

from bujo.models.base import BaseNocoDB

logger = logging.getLogger(__name__)


class PriceAlerts(BaseNocoDB):
    def create(self, ticker: str, direction: str, target_price: float, action: str = "") -> Optional[Dict[str, Any]]:
        data = {"Ticker": ticker, "Direction": direction, "TargetPrice": target_price}
        if action:
            data["Action"] = action
        response = requests.post(self._url(), json=data, headers=self.headers)
        if response.ok:
            return response.json()
        logger.error("PriceAlerts create failed: %s %s", response.status_code, response.text)
        return None

    def list_active(self) -> List[Dict[str, Any]]:
        return self._paginated_list({})

    def update(self, alert_id: int, **fields) -> bool:
        response = requests.patch(
            self._url(),
            json=[{"Id": alert_id, **fields}],
            headers=self.headers,
        )
        if not response.ok:
            logger.error("PriceAlerts update failed: %s %s", response.status_code, response.text)
        return response.ok
