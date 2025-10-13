"""
Contact Manager - Starter Template
Session 4: Functions and Data Structures
Anna Smirnova, October 2025

This is your starting point! Fill in the TODOs to build a working contact manager.
Try to implement these yourself before looking at solutions.
"""

import json

# ============================================================================
# INITIALISATION
# ============================================================================

contacts = []

# ============================================================================
# FEATURE 1: ADD CONTACTS
# ============================================================================

def add_contact(contacts_list, name, phone, email=""):
    """Add a new contact to our list"""
    contact = {
        "id": len(contacts_list) + 1,
        "name": name,
        "phone": phone,
        "email": email
    }

    contacts_list.append(contact)
    print(f"✓ Added {name} to contacts")
    return contact


def contact_exists(contacts_list, name):
    """Check if a contact already exists"""
    for contact in contacts_list:
        if contact["name"].lower() == name.lower():
            return True
    return False


def add_contact_safe(contacts_list, name, phone, email=""):
    """Add contact with duplicate prevention"""
    if contact_exists(contacts_list, name):
        print(f"⚠️  Contact '{name}' already exists.")
        return None
    else:
        return add_contact(contacts_list, name, phone, email)


# ============================================================================
# FEATURE 2: SEARCH CONTACTS
# ============================================================================

def search_contacts(contacts_list, search_term):
    """Find contacts by name or phone"""
    results = []
    search_term = search_term.lower()
    for contact in contacts_list:
        if (search_term in contact["name"].lower()) or (search_term in contact["phone"]):
            results.append(contact)
    return results


def display_search_results(contacts_list, search_term):
    """Display search results in a user-friendly way"""
    results = search_contacts(contacts_list, search_term)
    if not results:
        print(f"No contacts found for '{search_term}'.")
    else:
        print(f"Found {len(results)} contact(s):")
        for c in results:
            print(f"  • {c['name']}: {c['phone']}")


# ============================================================================
# FEATURE 3: DISPLAY ALL CONTACTS
# ============================================================================

def display_all_contacts(contacts_list):
    """Display all contacts in a formatted way"""
    if not contacts_list:
        print("No contacts to display.")
        return

    print("\nID  Name                 Phone           Email")
    print("--  -------------------  --------------  ---------------------")
    for c in contacts_list:
        print(f"{c['id']:<3} {c['name']:<20} {c['phone']:<14} {c['email']}")


# ============================================================================
# FEATURE 4: SAVE/LOAD
# ============================================================================

def save_contacts(contacts_list, filename="contacts.json"):
    """Save contacts to a file"""
    try:
        with open(filename, "w") as f:
            json.dump(contacts_list, f, indent=2)
        print(f"💾 Contacts saved to '{filename}'.")
        return True
    except Exception as e:
        print(f"❌ Error saving: {e}")
        return False


def load_contacts(filename="contacts.json"):
    """Load contacts from a file"""
    try:
        with open(filename, "r") as f:
            data = json.load(f)
        print(f"📂 Loaded {len(data)} contact(s) from '{filename}'.")
        return data
    except FileNotFoundError:
        print("⚠️  No saved contacts found.")
        return []
    except Exception as e:
        print(f"❌ Error loading: {e}")
        return []


# ============================================================================
# FEATURE 5: DELETE CONTACT
# ============================================================================

def delete_contact(contacts_list, name):
    """Delete a contact by name"""
    for i, contact in enumerate(contacts_list):
        if contact["name"].lower() == name.lower():
            removed = contacts_list.pop(i)
            print(f"🗑️  Deleted contact '{removed['name']}'.")
            return removed
    print("Contact not found.")
    return None


# ============================================================================
# FEATURE 6: STATISTICS
# ============================================================================

def get_contact_stats(contacts_list):
    """Get interesting statistics about contacts"""
    stats = {
        "total": len(contacts_list),
        "with_email": 0,
        "without_email": 0
    }

    for contact in contacts_list:
        if contact["email"]:
            stats["with_email"] += 1
        else:
            stats["without_email"] += 1

    return stats


# ============================================================================
# MENU SYSTEM
# ============================================================================

def display_menu():
    """Display the main menu"""
    print("\n" + "="*40)
    print("     CONTACT MANAGER MENU")
    print("="*40)
    print("1. Add contact")
    print("2. Search contacts")
    print("3. Display all contacts")
    print("4. Show statistics")
    print("5. Save contacts")
    print("6. Load contacts")
    print("7. Delete contact")
    print("0. Exit")
    print("="*40)


def run_contact_manager():
    """Main program loop"""
    contacts_list = load_contacts()

    while True:
        display_menu()
        choice = input("Enter your choice: ").strip()

        if choice == "0":
            save_contacts(contacts_list)
            print("👋 Goodbye! Contacts saved.")
            break

        elif choice == "1":
            name = input("Name: ").strip()
            phone = input("Phone: ").strip()
            email = input("Email (optional): ").strip()
            add_contact_safe(contacts_list, name, phone, email)

        elif choice == "2":
            term = input("Search by name or phone: ").strip()
            display_search_results(contacts_list, term)

        elif choice == "3":
            display_all_contacts(contacts_list)

        elif choice == "4":
            stats = get_contact_stats(contacts_list)
            print(f"\n📊 Total contacts: {stats['total']}")
            print(f"📧 With email: {stats['with_email']}")
            print(f"📴 Without email: {stats['without_email']}")

        elif choice == "5":
            save_contacts(contacts_list)

        elif choice == "6":
            contacts_list = load_contacts()

        elif choice == "7":
            name = input("Enter name to delete: ").strip()
            delete_contact(contacts_list, name)

        else:
            print("❌ Invalid choice, please try again.")


# ============================================================================
# BONUS: RECURSION EXAMPLE
# ============================================================================

def find_contact_recursive(contacts_list, name, index=0):
    """Find a contact using recursion (just for learning!)"""
    if index >= len(contacts_list):
        return None
    if contacts_list[index]["name"].lower() == name.lower():
        return contacts_list[index]
    return find_contact_recursive(contacts_list, name, index + 1)


# ============================================================================
# TEST / ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("📇 Contact Manager v1.0")
    print("=" * 40)
    run_contact_manager()
