import json
import time
import base64
import os
from datetime import datetime, timedelta
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

def load_cookies(filepath):
    """Reads cookies from a JSON file and handles potential errors."""
    try:
        with open(filepath, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: Could not find '{filepath}'.")
        return None
    except json.JSONDecodeError:
        print(f"Error: '{filepath}' is not valid JSON.")
        return None

def generate_date_range(start_date, end_date):
    """Generates a list of string dates between start and end (inclusive)."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    delta = end - start
    
    if delta.days < 0:
        print("Error: End date is before start date.")
        return []
        
    return [(start + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(delta.days + 1)]

def download_epaper_range():
    # 1. Get User Input
    print("--- Divya Bhaskar Bulk E-Paper Scraper ---")
    edition_name = input("Enter edition name (e.g., ahmedabad, kutch-bhaskar): ").strip()
    edition_id = input("Enter edition ID (e.g., 12, 546): ").strip()
    start_date = input("Enter start date (YYYY-MM-DD): ").strip()
    end_date = input("Enter end date (YYYY-MM-DD): ").strip()

    dates_to_scrape = generate_date_range(start_date, end_date)
    if not dates_to_scrape:
        return

    print(f"\nFound {len(dates_to_scrape)} days to process.")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()

        # 2. Load Session
        cookie_file = 'db_session.json'
        cookies = load_cookies(cookie_file)

        if not cookies:
            print("Exiting. Invalid or missing session cookies.")
            browser.close()
            return

        context.add_cookies(cookies)
        page = context.new_page()

        # 3. Iterate through each date in the range
        for current_date in dates_to_scrape:
            print(f"\n{'='*50}")
            print(f"Processing Date: {current_date}")
            print(f"{'='*50}")

            # Dynamically construct the URL
            target_url = f"https://www.divyabhaskar.co.in/epaper/detail-page/{edition_name}/{edition_id}/{current_date}"
            
            # Construct the nested folder structure: root/date/edition/
            root_dir = "epaper_downloads"
            save_dir = os.path.join(root_dir, current_date, edition_name)
            os.makedirs(save_dir, exist_ok=True)

            print(f"Navigating to: {target_url}")
            page.goto(target_url)
            page.wait_for_load_state("networkidle")

            try:
                # We use a try block here because if a paper doesn't exist for this date 
                # (e.g., holiday), the dropdown won't exist and we should skip to the next day.
                
                print("Opening dropdown menu...")
                # Using the explicit selector you confirmed worked previously
                dropdown_button = page.locator("#__next > div.BaseLayout_component_container__LaU36 > div.Navbar_component_container__8_RKl > ul > div > ul > li:nth-child(2) > div > span > span")
                
                # Wait for the button to ensure the page is actually an e-paper page
                dropdown_button.wait_for(state="visible", timeout=10000)
                dropdown_button.click()
                
                # Wait for the list to appear
                page.wait_for_selector("ul[class*='DetailNavbar_component_dropdown']", timeout=5000)

                # Extract all absolute page URLs
                page_links = page.locator("ul[class*='DetailNavbar_component_dropdown'] a").evaluate_all(
                    "elements => elements.map(el => el.href)"
                )
                
                total_pages = len(page_links)
                print(f"Found {total_pages} pages for {current_date}. Starting download...")

            except PlaywrightTimeoutError:
                print(f"[SKIP] No e-paper found for {current_date} or page structure changed.")
                continue # Skips to the next date in the loop

            # 4. Loop through the extracted URLs for this specific date
            for i, link in enumerate(page_links):
                print(f"  -> Fetching page {i + 1}/{total_pages}...")
                
                page.goto(link)
                page.wait_for_load_state("networkidle")
                time.sleep(1) # Small buffer

                try:
                    # Locate and extract the blob image
                    img_locator = page.locator("img[src^='blob:']")
                    img_locator.wait_for(state="visible", timeout=10000)
                    blob_url = img_locator.get_attribute("src")

                    # Execute JavaScript to fetch the blob and convert to Base64
                    base64_data_url = page.evaluate("""async (url) => {
                        const response = await fetch(url);
                        const blob = await response.blob();
                        return new Promise((resolve, reject) => {
                            const reader = new FileReader();
                            reader.onloadend = () => resolve(reader.result);
                            reader.onerror = reject;
                            reader.readAsDataURL(blob);
                        });
                    }""", blob_url)

                    # Decode the Base64 string
                    metadata, encoded_data = base64_data_url.split(",", 1)

                    # Save systematically
                    file_name = f"page_{i + 1:03d}.jpg"
                    file_path = os.path.join(save_dir, file_name)

                    with open(file_path, "wb") as file:
                        file.write(base64.b64decode(encoded_data))

                    print(f"     [SUCCESS] Saved {file_name}")

                except Exception as e:
                    print(f"     [ERROR] Failed to extract page {i + 1}: {e}")

                # Sleep to prevent bans
                time.sleep(3.5)

            print(f"Completed processing for {current_date}.")

        print("\nAll dates processed successfully.")
        browser.close()

if __name__ == "__main__":
    download_epaper_range()