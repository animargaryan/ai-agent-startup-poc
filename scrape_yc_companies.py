import json
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from tqdm import tqdm


def setup_driver():
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')  # run without opening a browser
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)


def scrape_yc_companies(max_companies=50):
    url = "https://www.ycombinator.com/companies"
    driver = setup_driver()
    driver.get(url)
    time.sleep(3)  # allow JS to load

    # Scroll down to load more companies
    last_height = driver.execute_script("return document.body.scrollHeight")
    while True and len(driver.find_elements(By.CLASS_NAME, "DirectoryListItem__Wrapper-sc-1tx4v4i-0")) < max_companies:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

    # Extract cards
    company_cards = driver.find_elements(By.CLASS_NAME, "_company_i9oky_355")
    data = []

    for card in tqdm(company_cards[:max_companies]):
        try:
            name = card.find_element(By.CLASS_NAME, "_coName_i9oky_470").text.strip()
            tagline = card.find_element(By.CLASS_NAME, "_coDescription_i9oky_495").text.strip()
            link = card.get_attribute("href")

            # Navigate to company page to get industry + description
            driver.execute_script("window.open(arguments[0]);", link)
            driver.switch_to.window(driver.window_handles[1])
            time.sleep(1)

            try:
                industry_el = driver.find_element(By.XPATH, "//a[contains(@href, '/companies/industry/')][1]")
                industry = industry_el.text.strip()
            except:
                industry = ""

            try:
                description_el = driver.find_element(By.XPATH, "//div[contains(@class, 'whitespace-pre-line')][1]")
                description = description_el.text.strip()
            except:
                description = tagline  # fallback

            driver.close()
            driver.switch_to.window(driver.window_handles[0])

            data.append({
                "name": name,
                "tagline": tagline,
                "industry": industry,
                "description": description
            })
        except Exception as e:
            print(f"Error extracting company: {e}")
            continue

    driver.quit()

    with open("yc_startups.json", "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved {len(data)} companies to yc_startups.json")


if __name__ == "__main__":
    scrape_yc_companies(max_companies=50)
