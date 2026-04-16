# File: returnto_main.py

def exit_prompt():
    response = input("Are you sure you want to exit? (y/n): ").strip().lower()
    if response == 'y':
        print("Exiting the program...")
        exit(0)
    elif response == 'n':
        print("Returning to the main menu.")
        return  # This returns control back to the main() loop
    else:
        print("Invalid input. Please enter 'y' or 'n'.")
        exit_prompt() # Recursive call if input is bad