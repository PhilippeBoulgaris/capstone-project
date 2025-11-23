"""
Contact Manager v2.0 - OOP Starter Template
Session 6: Object-Oriented Programming
Anna Smirnova, October 2025

Transform your Week 4 functional Contact Manager into an OOP version!

INSTRUCTIONS:
1. Complete the Contact class (optional but recommended)
2. Complete the ContactManager class methods
3. Test your implementation with the provided test code
"""

import json
from datetime import datetime
from typing import List, Dict, Optional


# ============================================================================
# CONTACT CLASS (Optional Challenge)
# ============================================================================

class Contact:
    """Represents a single contact with validation"""

    def __init__(self, name: str, phone: str, email: str = ""):
        """
        Initialize a new contact.

        TODO: Store name, phone, email as instance attributes
        TODO: Add a created_at timestamp using datetime.now().isoformat()
        """
        # TODO: Implement this
        pass

    def matches_search(self, search_term: str) -> bool:
        """
        Check if this contact matches a search term.

        TODO: Return True if search_term (case-insensitive) is found in
              either the name or phone number
        """
        # TODO: Implement this
        pass

    def to_dict(self) -> dict:
        """
        Convert contact to dictionary for JSON serialization.

        TODO: Return a dictionary with name, phone, email, created_at
        """
        # TODO: Implement this
        pass

    @classmethod
    def from_dict(cls, data: dict):
        """
        Create Contact from dictionary (for loading from JSON).

        TODO: Create a Contact instance from the dictionary
        HINT: Use cls(name, phone, email) to create instance
        """
        # TODO: Implement this
        pass

    def __str__(self):
        """
        String representation for display.

        TODO: Return a nice string like "Alice Smith: 555-0001 (alice@email.com)"
        HINT: Only show email in parentheses if it exists
        """
        # TODO: Implement this
        pass


# ============================================================================
# CONTACT MANAGER CLASS - Main Application
# ============================================================================

class ContactManager:
    """
    Manages a collection of contacts with save/load functionality.

    This is the OOP version of our Week 4 Contact Manager!
    Notice how the data (contacts list) and the functions that operate
    on it are now bundled together in a class.
    """

    def __init__(self):
        """
        Initialize with empty contact list.

        TODO: Create an empty list to store Contact objects
        HINT: self.contacts = []
        """
        # TODO: Implement this
        pass

    # ========================================================================
    # CORE FUNCTIONALITY
    # ========================================================================

    def add_contact(self, name: str, phone: str, email: str = "") -> Optional[Contact]:
        """
        Add a new contact to the manager.

        Week 4 version: add_contact(contacts_list, name, phone, email)
        Week 6 version: manager.add_contact(name, phone, email)

        TODO:
        1. Check if contact already exists using self.contact_exists()
        2. If exists, print warning and return None
        3. Create a new Contact object
        4. Append it to self.contacts
        5. Print success message
        6. Return the contact
        """
        # TODO: Implement this
        pass

    def contact_exists(self, name: str) -> bool:
        """
        Check if a contact with this name already exists.

        TODO: Loop through self.contacts and check if any contact
              has the same name (case-insensitive)
        """
        # TODO: Implement this
        pass

    def search(self, search_term: str) -> List[Contact]:
        """
        Find contacts matching the search term.

        Week 4: search_contacts(contacts_list, term)
        Week 6: manager.search(term)

        TODO: Return a list of contacts where the search_term matches
              either the name or phone
        HINT: Use the Contact's matches_search() method
        HINT: List comprehension makes this easy!
        """
        # TODO: Implement this
        pass

    def delete_contact(self, name: str) -> Optional[Contact]:
        """
        Delete a contact by name.

        TODO:
        1. Loop through self.contacts with enumerate()
        2. Find contact with matching name (case-insensitive)
        3. Use pop(i) to remove and return it
        4. Print success message
        5. Return the removed contact
        6. If not found, print message and return None
        """
        # TODO: Implement this
        pass

    def get_all_contacts(self) -> List[Contact]:
        """
        Return all contacts.

        TODO: Return a copy of the contacts list
        HINT: Use .copy() to avoid external modification
        """
        # TODO: Implement this
        pass

    # ========================================================================
    # FILE OPERATIONS
    # ========================================================================

    def save(self, filename: str = "contacts.json") -> bool:
        """
        Save contacts to a JSON file.

        Week 4: save_contacts(contacts_list, filename)
        Week 6: manager.save(filename)

        TODO:
        1. Convert each Contact object to a dictionary using to_dict()
        2. Use json.dump() to save the list of dictionaries
        3. Handle exceptions with try/except
        4. Print success/error messages
        5. Return True on success, False on error
        """
        # TODO: Implement this
        pass

    def load(self, filename: str = "contacts.json") -> bool:
        """
        Load contacts from a JSON file.

        Week 4: contacts = load_contacts(filename)
        Week 6: manager.load(filename)

        TODO:
        1. Use json.load() to read the file
        2. Convert each dictionary to a Contact using Contact.from_dict()
        3. Store in self.contacts
        4. Handle FileNotFoundError separately from other exceptions
        5. Print success/error messages
        6. Return True on success, False on error
        """
        # TODO: Implement this
        pass

    # ========================================================================
    # STATISTICS & UTILITY
    # ========================================================================

    def get_stats(self) -> dict:
        """
        Get statistics about the contact collection.

        TODO: Return a dictionary with:
        - "total": total number of contacts
        - "with_email": count of contacts with email
        - "without_email": count of contacts without email
        - "by_area_code": dict mapping area code (first 3 digits) to list of names

        HINT: Use sum() with a generator expression for counting
        """
        # TODO: Implement this
        pass

    def display_all(self):
        """
        Display all contacts in a formatted table.

        TODO:
        1. Check if list is empty, print message if so
        2. Print a table header with Name, Phone, Email columns
        3. Loop through contacts and print each one
        4. Use string formatting to align columns nicely
        """
        # TODO: Implement this
        pass

    def __len__(self):
        """
        Allow len(manager) to get contact count.

        TODO: Return the length of self.contacts
        """
        # TODO: Implement this
        pass

    def __str__(self):
        """
        String representation.

        TODO: Return something like "ContactManager with 5 contacts"
        """
        # TODO: Implement this
        pass


# ============================================================================
# TESTING CODE - Use this to test your implementation!
# ============================================================================

if __name__ == "__main__":
    print("Testing OOP Contact Manager\n")
    print("="*60)

    # Create a manager
    manager = ContactManager()
    print(f"Created: {manager}")

    # Test adding contacts
    print("\n--- Testing add_contact() ---")
    manager.add_contact("Alice Smith", "555-0001", "alice@email.com")
    manager.add_contact("Bob Jones", "555-0002")
    manager.add_contact("Charlie Brown", "555-0003", "charlie@email.com")

    # Test duplicate prevention
    print("\n--- Testing duplicate prevention ---")
    manager.add_contact("Alice Smith", "555-9999")  # Should fail

    # Test search
    print("\n--- Testing search() ---")
    results = manager.search("alice")
    print(f"Search for 'alice': {len(results)} result(s)")
    for contact in results:
        print(f"  Found: {contact}")

    # Test display
    print("\n--- Testing display_all() ---")
    manager.display_all()

    # Test stats
    print("\n--- Testing get_stats() ---")
    stats = manager.get_stats()
    print(f"Total: {stats['total']}")
    print(f"With email: {stats['with_email']}")
    print(f"Without email: {stats['without_email']}")

    # Test save
    print("\n--- Testing save() ---")
    manager.save("test_contacts.json")

    # Test load
    print("\n--- Testing load() ---")
    new_manager = ContactManager()
    new_manager.load("test_contacts.json")
    print(f"Loaded: {new_manager}")
    print(f"Contact count: {len(new_manager)}")

    # Test delete
    print("\n--- Testing delete_contact() ---")
    manager.delete_contact("Bob Jones")
    print(f"After deletion: {len(manager)} contacts")

    print("\n" + "="*60)
    print("If all tests printed expected output, you're done!")
    print("Expected output:")
    print("  - 3 contacts added successfully")
    print("  - 1 duplicate rejected")
    print("  - Search finds Alice")
    print("  - Stats show 2 with email, 1 without")
    print("  - Save/load preserves all data")
    print("  - Delete removes Bob")
