"""searching for hotels, booking and cancelling hotel and updating check-in and check-out dates"""
import sqlite3
from datetime import date, datetime
from typing import Optional, Union, Type
from langchain_core.tools import BaseTool
from langchain.pydantic_v1 import BaseModel, Field


class SearchHotelIntput(BaseModel):
    location: Optional[str] = Field(description="The location of the hotel.")
    name: Optional[str] = Field(description="The name of the hotel.")
    price_tier: Optional[str] = Field(
        description="The price tier of the hotel. Defaults to None. Examples: Midscale, Upper Midscale, Upscale, Luxury")
    checkin_date: Optional[Union[datetime, date]] = Field(description="The check-in date of the hotel.")
    checkout_date: Optional[Union[datetime, date]] = Field(description="The check-out date of the hotel.")


class SearchHotel(BaseTool):
    name: str = "search_hotels"
    description: str = "Search for hotels based on location, name, price tier, check-in date, and check-out date.."
    args_schema: Type[BaseModel] = SearchHotelIntput
    return_direct: bool = False

    db: str = ""

    def __init__(self, db):
        super().__init__()
        self.db = db

    def _run(self,
             location: Optional[str] = None,
             name: Optional[str] = None,
             price_tier: Optional[str] = None,
             checkin_date: Optional[Union[datetime, date]] = None,
             checkout_date: Optional[Union[datetime, date]] = None,
             ) -> list[dict]:
        """
        Search for hotels based on location, name, price tier, check-in date, and check-out date.

        Args:
            location (Optional[str]): The location of the hotel. Defaults to None.
            name (Optional[str]): The name of the hotel. Defaults to None.
            price_tier (Optional[str]): The price tier of the hotel. Defaults to None. Examples: Midscale, Upper Midscale, Upscale, Luxury
            checkin_date (Optional[Union[datetime, date]]): The check-in date of the hotel. Defaults to None.
            checkout_date (Optional[Union[datetime, date]]): The check-out date of the hotel. Defaults to None.

        Returns:
            list[dict]: A list of hotel dictionaries matching the search criteria.
        """
        conn = sqlite3.connect(self.db)
        cursor = conn.cursor()

        query = "SELECT * FROM hotels WHERE 1=1"
        params = []

        if location:
            query += " AND location LIKE ?"
            params.append(f"%{location}%")
        if name:
            query += " AND name LIKE ?"
            params.append(f"%{name}%")
        # For the sake of this tutorial, we will let you match on any dates and price tier.
        cursor.execute(query, params)
        results = cursor.fetchall()

        conn.close()

        return [
            dict(zip([column[0] for column in cursor.description], row)) for row in results
        ]

    @staticmethod
    def get_file_docstring():
        return get_tool_group_desc()


class BookHotelIntput(BaseModel):
    hotel_id: int = Field(description="The ID of the hotel to book.")


class BookHotel(BaseTool):
    name: str = "book_hotel"
    description: str = "Book a hotel by its ID."
    args_schema: Type[BaseModel] = BookHotelIntput
    return_direct: bool = False

    db: str = ""

    def __init__(self, db):
        super().__init__()
        self.db = db

    def _run(self, hotel_id: int) -> str:
        """
        Book a hotel by its ID.

        Args:
            hotel_id (int): The ID of the hotel to book.

        Returns:
            str: A message indicating whether the hotel was successfully booked or not.
        """
        conn = sqlite3.connect(self.db)
        cursor = conn.cursor()

        cursor.execute("UPDATE hotels SET booked = 1 WHERE id = ?", (hotel_id,))
        conn.commit()

        if cursor.rowcount > 0:
            conn.close()
            return f"Hotel {hotel_id} successfully booked."
        else:
            conn.close()
            return f"No hotel found with ID {hotel_id}."

    @staticmethod
    def get_file_docstring():
        return get_tool_group_desc()


class UpdateHotelIntput(BaseModel):
    hotel_id: int = Field(description="The ID of the hotel to update.")
    checkin_date: Optional[Union[datetime, date]] = Field(description="The new check-in date of the hotel.")
    checkout_date: Optional[Union[datetime, date]] = Field(description="The new check-out date of the hotel.")


class UpdateHotel(BaseTool):
    name: str = "update_hotel"
    description: str = "Update a hotel's check-in and check-out dates by its ID."
    args_schema: Type[BaseModel] = UpdateHotelIntput
    return_direct: bool = False

    db: str = ""

    def __init__(self, db):
        super().__init__()
        self.db = db

    def _run(self,
             hotel_id: int,
             checkin_date: Optional[Union[datetime, date]] = None,
             checkout_date: Optional[Union[datetime, date]] = None,
             ) -> str:
        """
        Update a hotel's check-in and check-out dates by its ID.

        Args:
            hotel_id (int): The ID of the hotel to update.
            checkin_date (Optional[Union[datetime, date]]): The new check-in date of the hotel. Defaults to None.
            checkout_date (Optional[Union[datetime, date]]): The new check-out date of the hotel. Defaults to None.

        Returns:
            str: A message indicating whether the hotel was successfully updated or not.
        """
        conn = sqlite3.connect(self.db)
        cursor = conn.cursor()

        if checkin_date:
            cursor.execute(
                "UPDATE hotels SET checkin_date = ? WHERE id = ?", (checkin_date, hotel_id)
            )
        if checkout_date:
            cursor.execute(
                "UPDATE hotels SET checkout_date = ? WHERE id = ?",
                (checkout_date, hotel_id),
            )

        conn.commit()

        if cursor.rowcount > 0:
            conn.close()
            return f"Hotel {hotel_id} successfully updated."
        else:
            conn.close()
            return f"No hotel found with ID {hotel_id}."

    @staticmethod
    def get_file_docstring():
        return get_tool_group_desc()


class CancelHotelIntput(BaseModel):
    hotel_id: int = Field(description="The ID of the hotel to cancel.")


class CancelHotel(BaseTool):
    name: str = "cancel_hotel"
    description: str = "Cancel a hotel by its ID."
    args_schema: Type[BaseModel] = CancelHotelIntput
    return_direct: bool = False

    db: str = ""

    def __init__(self, db):
        super().__init__()
        self.db = db

    def _run(self, hotel_id: int) -> str:
        """
        Cancel a hotel by its ID.

        Args:
            hotel_id (int): The ID of the hotel to cancel.

        Returns:
            str: A message indicating whether the hotel was successfully cancelled or not.
        """
        conn = sqlite3.connect(self.db)
        cursor = conn.cursor()

        cursor.execute("UPDATE hotels SET booked = 0 WHERE id = ?", (hotel_id,))
        conn.commit()

        if cursor.rowcount > 0:
            conn.close()
            return f"Hotel {hotel_id} successfully cancelled."
        else:
            conn.close()
            return f"No hotel found with ID {hotel_id}."

    @staticmethod
    def get_file_docstring():
        return get_tool_group_desc()


def get_tool_group_desc():
    return __doc__
