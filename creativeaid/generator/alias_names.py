import pywikibot


def alias_names(word):
    site = pywikibot.Site("en", "wikipedia")
    page = pywikibot.Page(site, word)
    item = pywikibot.ItemPage.fromPage(page)

    item_dict = item.get()  # Get the item dictionary
    return item_dict['aliases']['en']  # Get the claim dictionary


if __name__ == '__main__':
    print(alias_names("Barack Obama"))
