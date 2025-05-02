# 2024-10-29
from general import *


def get_etu_difference(df_arts, df_etus, save_excel=False):
    df_etu_check = df_arts[['PRODUCT', 'ETU', 'PRODUCT_NAME']]\
        .rename(
            columns={
                'ETU': 'etu_from_arts',
            }
        )\
        .merge(
            df_etus[['PRODUCT', 'ETU']]\
                .rename(
                    columns={                    
                        'PRODUCT': 'PRODUCT',
                        'ETU': 'etu_from_etu',
                    }
                ),
            how='inner'
        ).drop_duplicates()

    filter_ = df_etu_check['etu_from_arts'] != df_etu_check['etu_from_etu']
    columns = ['PRODUCT', 'PRODUCT_NAME', 'etu_from_arts', 'etu_from_etu']
    df_etu_difference = df_etu_check[filter_].dropna()[columns].sort_values('PRODUCT').reset_index(drop=True)
    if save_excel:
        path = '../../misc/etu_difference.xlsx'
        df_etu_difference.to_excel(path, index=False)
    return df_etu_difference


def get_double_etus_arts(df_arts, products):
    filter_ = df_arts['PRODUCT'].isin(products)
    count_etus_arts = df_arts[filter_][['PRODUCT', 'ETU']]\
        .drop_duplicates()\
        .groupby('PRODUCT')\
        .count()
    filter_ = count_etus_arts['ETU'] > 1
    double_etus_arts = set(count_etus_arts[filter_].index)
    return double_etus_arts


def get_double_etus_etus(df_etus, products):
    filter_ = df_etus['PRODUCT'].isin(products)
    count_etus_etus = df_etus[filter_][['PRODUCT', 'ETU']]\
        .drop_duplicates()\
        .groupby('PRODUCT')\
        .count()
    filter_ = count_etus_etus['ETU'] > 1
    double_etus_etus = set(count_etus_etus[filter_].index)
    return double_etus_etus


def filter_products_by_test_dates_local(products, test_days, config=CONFIG):
    products_remain = copy.deepcopy(products)
    for product in tqdm.tqdm(products):
        df_sales_product = sales_load_product(product, config=config)
        condition_has_test = df_sales_product['DATE'].isin(test_days)
        if df_sales_product[condition_has_test].empty:
            products_remain.remove(product)
    return products_remain


def filter_etus_by_test_dates_local(etus, test_days, config=CONFIG):
    etus_remain = copy.deepcopy(etus)
    for etu in tqdm.tqdm(etus):
        df_sales_etu = sales_load_etu(etu, config=config)
        condition_has_test = df_sales_etu['DATE'].isin(test_days)
        if df_sales_etu[condition_has_test].empty:
            etus_remain.remove(etu)
    return etus_remain


def filter_products_by_catalogs(products=None, df_assort=None, df_arts=None, df_etus=None, config=CONFIG):
    if df_assort is None:
        df_assort = process_assort(load_assort(config=config))
    if df_arts is None:
        df_arts = process_arts(load_arts(config=config))
    if df_etus is None:
        df_etus = process_etus(load_etus(config=config))
    assort_active = get_assort_active(df_assort)
    if products is None:
        products = assort_active
    else:
        products = products.intersection(assort_active)
    products_arts = set(process_arts(df_arts)['PRODUCT'])
    products = products.intersection(products_arts)
    products = products.intersection(set(df_etus['PRODUCT']))
    # check if etu in arts differs from etu in etus table:
    df_etu_difference = get_etu_difference(df_arts, df_etus)
    products -= set(df_etu_difference['PRODUCT'])
    double_etus_arts = get_double_etus_arts(df_arts, products)
    products -= double_etus_arts
    double_etus_etus = get_double_etus_etus(df_etus, products)
    products -= double_etus_etus
    return products


def get_sums_products(products, sites):
    df_sums_products = pd.DataFrame(
        index=sorted(list(products)),
        columns=sorted(list(sites))
    )
    for product in tqdm.tqdm(products):
        df_sales_product = sales_load_product(product)
        df_sums_products.loc[product] = df_sales_product\
            .groupby('SITE')['SALES_QUANTITY']\
            .sum()
    return df_sums_products


def get_sums_etus(etus, sites):
    df_sums_etus = pd.DataFrame(
        index=sorted(list(etus)),
        columns=sorted(list(sites))
    )
    for etu in tqdm.tqdm(etus):
        df_sales_etu = sales_load_etu(etu)
        df_sums_etus.loc[etu] = df_sales_etu\
            .groupby('SITE')['SALES_QUANTITY']\
            .sum()
    return df_sums_etus


def get_etu_products(df_etus=None, products=None, config=CONFIG):
    if df_etus is None:
        df_etus = load_etus(config=config)
        df_etus = process_etus(df_etus, config=config)
    df_etu_products = df_etus.copy()[['PRODUCT', 'ETU']].drop_duplicates()
    if products is not None:
        filter_ = df_etu_products['PRODUCT'].isin(products)
        df_etu_products = df_etu_products[filter_]
    df_etu_products = df_etu_products.reset_index(drop=True)\
        .rename(columns={'PRODUCT': 'ETU_PRODUCTS'})\
        .groupby('ETU')\
        .agg(list)
    df_etu_products['N_PRODUCTS'] = df_etu_products['ETU_PRODUCTS'].apply(len)
    df_etu_products = df_etu_products.sort_values('N_PRODUCTS', ascending=False)
    return df_etu_products


def get_product_etus(df_arts=None, config=CONFIG):
    if df_arts is None:
        df_arts = load_arts(config=config)
        df_arts = process_arts(df_arts, config=config)
    df_product_etus = df_arts[['PRODUCT', 'ETU']]\
        .groupby(['PRODUCT'])\
        .agg(ETU = ('ETU', 'first'), N_ETUS = ('ETU', 'count'))
    return df_product_etus


def get_sums_etu_from_products(df_etus, etus=None, products=None, sites=None, config=CONFIG):
    if sites is None:
        sites = sales_get_sites_active(config=config)
    df_etu_products = get_etu_products(df_etus, products=products)
    if etus is None:
        etus = set(df_etu_products.index)
    df_sums_etu = pd.DataFrame(
        index=sorted(list(etus)),
        columns=sorted(list(sites))
    )
    for etu in tqdm.tqdm(etus):
        products_etu = df_etu_products.loc[etu, 'ETU_PRODUCTS']
        sales_etu = []
        for product in products_etu:
            df_sales_product = sales_load_product(product)
            sales_etu.append(df_sales_product[['SITE', 'SALES_QUANTITY']])        
        df_sums_etu.loc[etu] = pd\
            .concat(sales_etu)\
            .groupby('SITE')\
            .sum()['SALES_QUANTITY']
    return df_sums_etu


def etu_split_single_multi(df_etu_products):
    filter_ = df_etu_products['N_PRODUCTS'] > 1
    etus_single = set(df_etu_products[~filter_].index)
    etus_multi = set(df_etu_products[filter_].index)
    return etus_single, etus_multi


def get_etus_single_multi(df_etus, products=None):
    df_etu_products = get_etu_products(df_etus, products)
    etus_single, etus_multi = etu_split_single_multi(df_etu_products)
    return etus_single, etus_multi


def load_products_set_from_sales_files(config=CONFIG):
    pq_dir = '/'.join(
        [
            config['DATA_DIR'],
            config['SOURCES']['SALES']['FOLDER'],
            'products'
        ]
    )
    products = {i.replace('.pq', '') for i in os.listdir(pq_dir)}
    return products


def load_etus_set_from_sales_files(config=CONFIG):
    pq_dir = '/'.join(
        [
            config['DATA_DIR'],
            config['SOURCES']['SALES']['FOLDER'],
            'etus'
        ]
    )
    recent_folder = sorted(os.listdir(pq_dir))[-1]
    pq_dir = '/'.join(
        [
            pq_dir,
            recent_folder
        ]
    )
    etus = {i.replace('.pq', '') for i in os.listdir(pq_dir)}
    return etus


def load_sales_site_dates(config=CONFIG):
    sales_site_dates_path = '/'.join(
        [
            config['DATA_DIR'],
            'sales',
            'sales_site_dates.csv'
        ]
    )
    df_sales_site_dates = pd.read_csv(sales_site_dates_path)\
        .set_index('SITE')
    return df_sales_site_dates


def sales_get_sites_active(df_sales_site_dates=None, config=CONFIG):
    if df_sales_site_dates is None:
        df_sales_site_dates = load_sales_site_dates(config=config)
    date_max_common = df_sales_site_dates['date_max'].max()
    condition = df_sales_site_dates['date_max'] == date_max_common
    sites_active = set(df_sales_site_dates[condition].index)
    return sites_active


def process_sites(df_sites):
    conditions = ~df_sites['name'].str.contains('НЕ ИСП')
    conditions &= df_sites['formatname'] != 'РЦ'
    conditions &= df_sites['closedate'].isna()
    conditions &= ~df_sites['opendate'].isna()
    return df_sites[conditions].reset_index(drop=True)


def process_arts(df_arts, config=CONFIG):
    columns_dict = config['SOURCES']['PRODUCTS']['COLUMNS']
    columns = list(columns_dict.values())
    df_arts = df_arts.copy().rename(columns=columns_dict)[columns]
    category_index = df_arts['CATEGORY'].str.split('.').str[0].astype(int)
    conditions = [
        # df_arts['SEASONAL'] == 0,
        df_arts['PRODUCT_ACTIVE'] == 1,
        (category_index > 1) & (category_index < 20)
    ]
    conditions = np.all(conditions, axis=0)
    df_arts = df_arts[conditions]\
        .drop_duplicates()\
        .dropna()\
        .reset_index(drop=True)
    return df_arts


def process_assort(df_assort, config=CONFIG):
    columns_dict = config['SOURCES']['ASSORT']['COLUMNS']
    df_assort = df_assort\
        .copy()\
        .rename(columns=columns_dict)
    return df_assort\
        .groupby('PRODUCT')\
        .agg({'DATE_FROM': 'min', 'DATE_TO': 'max', 'SITE': 'unique'})


def process_etus(df_etus, config=CONFIG):
    columns_dict = config['SOURCES']['ETUS']['COLUMNS']
    columns = list(columns_dict.values())
    df_etus = df_etus\
        .copy()\
        .rename(columns=columns_dict)[columns]\
        .drop_duplicates()\
        .dropna()
    conditions = [
        df_etus['ETU_ACTIVE'] == 1,
        ~df_etus['ETU_NAME'].str.contains('in out'),
        ~df_etus['ETU_NAME'].str.contains('In out'),
        ~df_etus['ETU_NAME'].str.contains('ЕТУ')
    ]
    filter_ = np.all(conditions, axis=0)
    df_etus = df_etus[filter_].reset_index(drop=True)
    return df_etus


def load_catalog(catalog_name, config=CONFIG):
    catalog_dir = '/'.join(
        [
            config['DATA_DIR'],
            catalog_name
        ]
    )
    catalog_filename = sorted(os.listdir(catalog_dir))[-1]
    catalog_path = '/'.join(
        [
            catalog_dir,
            catalog_filename
        ]
    )
    df_catalog = pd.read_parquet(catalog_path)
    msg = f'Loaded {catalog_name} from {catalog_path}.'
    logger.info(msg)
    return df_catalog


def load_assort(config=CONFIG):
    return load_catalog(catalog_name='assort_ordered', config=config)


def load_sites(config=CONFIG):
    return load_catalog(catalog_name='sites', config=config)


def load_arts(config=CONFIG):
    return load_catalog(catalog_name='arts', config=config)


def load_etus(config=CONFIG):
    return load_catalog(catalog_name='etus', config=config)


def get_assort_ceased(df_assort=None, config=CONFIG):
    if df_assort is None:
        df_assort = process_assort(load_assort(config=config))
    return set(
        df_assort[df_assort['DATE_TO'] < df_assort['DATE_TO'].max()]\
            .index\
            .to_list()
    )


def get_assort_active(df_assort=None, config=CONFIG):
    if df_assort is None:
        df_assort = process_assort(load_assort(config=config))
    return set(
        df_assort[df_assort['DATE_TO'] == df_assort['DATE_TO'].max()]\
            .index\
            .to_list()
    )


def read_sites_dir(config=CONFIG):
    pq_dir = '/'.join(
        [
            config['DATA_DIR'],
            config['SOURCES']['SALES']['FOLDER']
        ]
    )
    sites_list = sorted([i.replace('.pq', '') for i in os.listdir(pq_dir)])
    return sites_list


def site_load_sales(site_id, config=CONFIG):
    pq_path = '/'.join(
        [
            config['DATA_DIR'],
            config['SOURCES']['SALES']['FOLDER'],
            f'{site_id}.pq'
        ]
    )
    df_sales_site = pd. \
        read_parquet(pq_path). \
        rename(columns=config['SOURCES']['SALES']['COLUMNS'])
    df_sales_site['DATE'] = pd.to_datetime(df_sales_site['DATE'])
    return df_sales_site


def sales_load_product_raw(product, config=CONFIG):
    pq_path = '/'.join(
        [
            config['DATA_DIR'],
            'products',
            'pq',
            f'{product}.pq'
        ]
    )
    columns_dict = config['SOURCES']['SALES']['COLUMNS']
    columns = list(columns_dict.values())
    df_sales_product = pd\
        .read_parquet(pq_path)\
        .dropna()\
        .rename(columns=columns_dict)
    df_sales_product = df_sales_product[columns]
    df_sales_product['DATE'] = pd.to_datetime(df_sales_product['DATE'])
    return df_sales_product


def sales_load_product(product, config=CONFIG):
    pq_path = '/'.join(
        [
            config['DATA_DIR'],
            config['SOURCES']['SALES']['FOLDER'],
            'products',
            f'{product}.pq'
        ]
    )
    df_sales_product = pd.read_parquet(pq_path)
    return df_sales_product


def sales_load_etu(etu, config=CONFIG):
    pq_dir = '/'.join(
        [
            config['DATA_DIR'],
            config['SOURCES']['SALES']['FOLDER'],
            'etus'
        ]
    )
    recent_folder = sorted(os.listdir(pq_dir))[-1]
    pq_dir = '/'.join(
        [
            pq_dir,
            recent_folder
        ]
    )
    pq_path = '/'.join(
        [
            pq_dir,
            f'{etu}.pq'
        ]
    )
    df_sales_product = pd.read_parquet(pq_path)
    return df_sales_product


def get_dates_sites(config=CONFIG) -> tuple:
    '''
    df_dates_sites, date_from_global, date_to_global = get_dates_sites(config=CONFIG)
    '''
    sites_list = read_sites_dir(config=config)
    dates_sites = {}
    for site in tqdm.tqdm(sites_list):
        df_sales_site = site_load_sales(site)
        date_from = df_sales_site['DATE'].min()
        date_to = df_sales_site['DATE'].max()
        dates_sites[site] = {'date_from': date_from, 'date_to': date_to}
    df_dates_sites = pd.DataFrame(dates_sites).T
    date_from_global = df_dates_sites['date_from'].min()  # might cause problems with inconsistent data
    date_to_global = df_dates_sites['date_to'].max()  # might cause problems with inconsistent data
    stall_days = (date_to_global - df_dates_sites['date_to']).dt.days
    df_dates_sites['stall_days'] = stall_days
    return df_dates_sites, date_from_global, date_to_global


def get_sites_active(
        df_dates_sites,
        write_inactive=True,
        config=CONFIG
) -> tuple:
    '''
    df_sites_active, df_sites_inactive = get_sites_active(
        write_inactive=True,
        config=CONFIG
    )
    '''
    df_sites_active = df_dates_sites \
        .query('stall_days <= 0') \
        .drop(columns='stall_days')
    df_sites_inactive = df_dates_sites \
        .query('stall_days > 0')
    if write_inactive:
        sites_inactive_path = '/'.join(
            [
                config['DATA_DIR'],
                config['SITES_INACTIVE']
            ]
        )
        df_sites_inactive.to_csv(sites_inactive_path)
    return df_sites_active, df_sites_inactive


def eliminate_sites_short(
        df_sites_active,
        write_csv=True,
        config=CONFIG
) -> tuple:
    '''
    df_sites_to_go, df_sites_short = eliminate_sites_short(
        df_sites_active,
        write_csv=True,
        config=CONFIG
    )
    '''
    time_spans = df_sites_active['date_to'] \
        - df_sites_active['date_from']
    is_short = time_spans.dt.days < config['DAYS_SHORT']
    df_sites_short = df_sites_active[is_short]
    df_sites_to_go = df_sites_active[~is_short]
    if write_csv:
        path_sites_short = '/'.join(
            [
                config['DATA_DIR'],
                config['SITES_SHORT']
            ]
        )
        path_sites_to_go = '/'.join(
            [
                config['DATA_DIR'],
                config['SITES_TO_GO']
            ]
        )
        df_sites_short.to_csv(path_sites_short)
        df_sites_to_go.to_csv(path_sites_to_go)
    return df_sites_to_go, df_sites_short


def refresh_sites_to_go(
        write_csv=True,
        config=CONFIG
) -> pd.DataFrame:
    '''
    df_sites_to_go = sites_to_go(
        write_csv=True,
        config=CONFIG
    )
    '''
    df_dates_sites, _, _ = get_dates_sites(config=config)
    df_sites_active, _ = get_sites_active(
        df_dates_sites,
        write_inactive=write_csv,
        config=config
    )
    df_sites_to_go, _ = eliminate_sites_short(
        df_sites_active,
        write_csv=write_csv,
        config=config
    )
    return df_sites_to_go


def load_sites_to_go(config=CONFIG) -> pd.DataFrame:
    '''
    df_sites_to_go = load_sites_to_go(config=CONFIG)
    '''
    path_sites_to_go = '/'.join(
        [
            config['DATA_DIR'],
            config['SITES_TO_GO']
        ]
    )
    if os.path.exists(path_sites_to_go):
        df_sites_to_go = pd.read_csv(path_sites_to_go)
        df_sites_to_go = df_sites_to_go \
            .rename(columns={'Unnamed: 0': 'ID_SYM'}) \
            .set_index('ID_SYM')
    else:
        df_sites_to_go = refresh_sites_to_go(
            write_csv=True,
            config=CONFIG
        )
        df_sites_to_go.index.name = 'ID_SYM'
    return df_sites_to_go


def refresh_dates_products_sites_recent(write_csv=True, config=CONFIG):
    products_dates_sites = {}
    sites = load_sites_to_go().index
    for site_id in tqdm.tqdm(sites):
        df_sales_site = site_load_sales(site_id)
        products = get_products_by_revenue(df_sales_site)
        dates_products = {}
        for product in tqdm.tqdm(products):
            df_sales_product = df_sales_site \
                .query('PRODUCT == @product') \
                .drop(columns='PRODUCT') \
                .reset_index(drop=True)
            date_to = df_sales_product['DATE'].max()
            dates_products[product] = date_to
        products_dates_sites[site_id] = dates_products
    df_products_dates_sites = pd.DataFrame(products_dates_sites)
    if write_csv:
        csv_path = '/'.join(
            [
                config['DATA_DIR'],
                config['DATES_PRODUCTS_SITES_RECENT']
            ]
        )
        df_products_dates_sites.to_csv(csv_path)
    return df_products_dates_sites


def refresh_dates_products_sites(write_csv=True, config=CONFIG) -> tuple:
    dates_products_sites_first = {}
    dates_products_sites_recent = {}
    sites = load_sites_to_go().index
    for site_id in tqdm.tqdm(sites):
        df_sales_site = site_load_sales(site_id)
        products = get_products_by_revenue(df_sales_site)
        dates_products_first = {}
        dates_products_recent = {}
        for product in tqdm.tqdm(products):
            df_sales_product = df_sales_site \
                .query('PRODUCT == @product') \
                .drop(columns='PRODUCT') \
                .reset_index(drop=True)
            date_from = df_sales_product['DATE'].min()
            date_to = df_sales_product['DATE'].max()
            dates_products_first[product] = date_from
            dates_products_recent[product] = date_to
        dates_products_sites_first[site_id] = dates_products_first
        dates_products_sites_recent[site_id] = dates_products_recent
    df_dates_products_sites_first = pd.DataFrame(dates_products_sites_first)
    df_dates_products_sites_recent = pd.DataFrame(dates_products_sites_recent)
    if write_csv:
        csv_path_first = '/'.join(
            [
                config['DATA_DIR'],
                config['DATES_PRODUCTS_SITES_FIRST']
            ]
        )
        df_dates_products_sites_first.to_csv(csv_path_first)
        csv_path_recent = '/'.join(
            [
                config['DATA_DIR'],
                config['DATES_PRODUCTS_SITES_RECENT']
            ]
        )
        df_dates_products_sites_recent.to_csv(csv_path_recent)
    return df_dates_products_sites_first, df_dates_products_sites_recent


def load_products_dates_sites(config=CONFIG) -> tuple:
    csv_path_first = '/'.join(
        [
            config['DATA_DIR'],
            config['DATES_PRODUCTS_SITES_FIRST']
        ]
    )
    csv_path_recent = '/'.join(
        [
            config['DATA_DIR'],
            config['DATES_PRODUCTS_SITES_RECENT']
        ]
    )
    if os.path.exists(csv_path_first) and os.path.exists(csv_path_recent):
        df_products_dates_sites_first = pd.read_csv(
            csv_path_first, low_memory=False
        ) \
            .rename(columns={'Unnamed: 0': 'PRODUCT'}) \
            .set_index('PRODUCT') \
            .astype('datetime64[ns]')
        df_products_dates_sites_recent = pd.read_csv(
            csv_path_recent, low_memory=False
        ) \
            .rename(columns={'Unnamed: 0': 'PRODUCT'}) \
            .set_index('PRODUCT') \
            .astype('datetime64[ns]')
    else:
        (
            df_products_dates_sites_first,
            df_products_dates_sites_recent
        ) = refresh_dates_products_sites(
            write_csv=True,
            config=config
        )
    return df_products_dates_sites_first, df_products_dates_sites_recent


def get_products_to_go(
        common_only: bool = True,
        config: dict = CONFIG
):
    _, df_products_dates_sites_recent = load_products_dates_sites(config=config)
    date_to_global = pd.Timestamp(config['DATE_UP_TO'])
    if common_only:
        recent_transactions = df_products_dates_sites_recent \
            .dropna() \
            .max(axis='columns')
    else:
        recent_transactions = df_products_dates_sites_recent \
            .max(axis='columns')
    df_recent_transactions = pd.DataFrame(
        recent_transactions,
        columns=['date_to']
    )
    stall_days_global = (date_to_global - recent_transactions).dt.days
    df_recent_transactions['stall_days'] = stall_days_global
    days_stall_limit = config['DAYS_STALL']
    return df_recent_transactions \
        .query('stall_days <= @days_stall_limit') \
        .index \
        .to_list()


def get_short_products(mode: str = 'max', config=CONFIG):
    '''
    mode (str): either 'min' or 'max', defaults to 'max'.
     With 'max' it is enough for a product to be present in a single site to be counted in.
    '''
    df_from, df_to = load_products_dates_sites(config=config)
    df_short = (df_to - df_from).dropna()
    if mode == 'min':
        df_short = df_short.min(axis='columns')
    else:
        df_short = df_short.max(axis='columns')
    df_short = df_short.dt.days <= config['DAYS_SHORT']
    short_products = set(df_short.index[df_short].to_list())
    return short_products


#####################максимальное количество нерабочих дней рейтинг############################

def get_dates_with_no_sales(site_df):
    """
    Функция для поиска дат, когда в магазине не было продаж.

    Аргументы:
    - site_df: DataFrame с данными по одному магазину.

    Возвращает:
    - Список дат, когда в магазине не было продаж.
    код для всех магазинов:
    no_sales_dict = {}
    for site in sites:
        site_df = site_load_sales(site)  # Загрузка данных для магазина
        site_df['ID_SYM'] = site  # Добавление идентификатора магазина
        no_sales_dates = get_dates_with_no_sales(site_df)
        no_sales_dict[site] = no_sales_dates
    """
    # Преобразование DATE в формат datetime, если это еще не сделано
    site_df['DATE'] = pd.to_datetime(site_df['DATE'])

    # Получаем полный диапазон дат между минимальной и максимальной датами в данных
    all_dates = pd.date_range(start=site_df['DATE'].min(), end=site_df['DATE'].max())

    # Находим уникальные даты с продажами
    sales_dates = site_df['DATE'].unique()

    # Вычисляем даты, когда не было продаж
    no_sales_dates = set(all_dates) - set(sales_dates)

    return sorted(no_sales_dates)


def top_no_sales_stores(no_sales_dict, threshold=1):
    """
    Функция для подсчета максимального количества последовательных дней простоя для каждого магазина
    и создания рейтинга магазинов по максимальной длине простоя.

    Аргументы:
    - no_sales_dict: Словарь, где ключи - магазины, значения - список дат, когда не было продаж.
    - threshold: Порог для минимальной длины простоя (по умолчанию 1 день).

    Возвращает:
    - sorted_stores: Отсортированный список кортежей (ID магазина, максимальная длина простоя).
    """
    # Создание обратного словаря: для каждой даты список магазинов, которые не имели продаж в эту дату
    date_to_sites_dict = defaultdict(list)
    for site, dates in no_sales_dict.items():
        for date in dates:
            date_to_sites_dict[date].append(site)

    # Преобразуем ключи (даты) в сортированный список
    sorted_dates = sorted(date_to_sites_dict.keys())

    # Словарь для хранения периодов простоя каждого магазина
    downtime_sequences = defaultdict(list)

    # Подсчет последовательных дней простоя
    for date in sorted_dates:
        stores = date_to_sites_dict[date]
        for store in stores:
            if downtime_sequences[store] and (downtime_sequences[store][-1][1] == date - pd.Timedelta(days=1)):
                # Продолжаем последовательность простоя
                downtime_sequences[store][-1] = (downtime_sequences[store][-1][0], date)
            else:
                # Начинаем новую последовательность простоя
                downtime_sequences[store].append((date, date))

    # Подсчет максимальной длины простоя для каждого магазина
    max_downtimes = {}
    for store, periods in downtime_sequences.items():
        max_duration = 0
        for start, end in periods:
            duration = (end - start).days + 1  # Включаем и стартовый день
            if duration >= threshold:
                max_duration = max(max_duration, duration)
        if max_duration >= threshold:
            max_downtimes[store] = max_duration

    # Создание рейтинга магазинов по максимальной длине простоя
    sorted_stores = sorted(max_downtimes.items(), key=lambda item: item[1], reverse=True)

    return sorted_stores


def identify_stores(df, target_year, target_month, new_store_months):
    """
    для отбора по всем магазинам
    closed_stores_total = []
    new_stores_total = []

    for site in sites:
        df = pd.DataFrame(site_load_sales(site))
        df['ID_SYM'] = site
        closed_stores, new_stores = identify_stores(df, target_year=2024,  target_month=3, new_store_months=12)
        closed_stores_total += closed_stores
        new_stores_total += new_stores


    """
    # Фильтруем данные до целевого месяца
    target_date = pd.Timestamp(year=target_year, month=target_month, day=1)
    data_until_target = df[df['DATE'] < target_date]

    # Фильтруем данные для целевого месяца
    data_in_target_month = df[(df['DATE'].dt.year == target_year) & (df['DATE'].dt.month == target_month)]

    # Магазины, которые были активны до целевого месяца
    active_before_target = data_until_target['ID_SYM'].unique()

    # Магазины, которые активны в целевом месяце
    active_in_target = data_in_target_month['ID_SYM'].unique()

    # Определяем закрытые магазины (были активны до целевого месяца, но не активны в нем)
    closed_stores = set(active_before_target) - set(active_in_target)

    # Фильтруем данные за последние new_store_months до целевого месяца
    new_store_start_date = target_date - pd.DateOffset(months=new_store_months)
    data_new_stores_period = df[(df['DATE'] >= new_store_start_date) & (df['DATE'] < target_date)]

    # Магазины, которые начали продажи в последние new_store_months
    active_in_new_period = data_new_stores_period['ID_SYM'].unique()

    # Определяем новые магазины (активны в целевом месяце, но не были активны до последних new_store_months)
    new_stores = set(active_in_target) - set(active_before_target) - set(active_in_new_period)

    return list(closed_stores), list(new_stores)


def filter_products_last_n_months(site_df, n_months, product_presence=None):
    """
    Функция для отбора товаров, которые продаются во всех магазинах последние N месяцев.

    Аргументы:
    - site_df: DataFrame с данными по одному магазину.
    - n_months: Количество месяцев для анализа.
    - product_presence: Множество товаров, продающихся во всех магазинах. Если None, будет инициализировано первыми данными.

    Возвращает:
    - Обновленное множество товаров, которые продаются во всех магазинах.
    """
    # Преобразование DATE в формат datetime, если это еще не сделано
    site_df['DATE'] = pd.to_datetime(site_df['DATE'])

    # Находим максимальную дату в датафрейме
    max_date = site_df['DATE'].max()

    # Определяем год и месяц максимальной даты
    max_year = max_date.year
    max_month = max_date.month

    # Создаем список последних N месяцев
    selected_months = []
    for i in range(n_months):
        month = max_month - i
        year = max_year
        if month <= 0:
            month += 12
            year -= 1
        selected_months.append((year, month))

    # Фильтрация данных по последним N месяцам
    site_df['year_month'] = site_df['DATE'].dt.to_period('M')
    selected_periods = [f"{year}-{month:02d}" for year, month in selected_months]
    site_filtered = site_df[site_df['year_month'].astype(str).isin(selected_periods)]

    # Определение товаров, продающихся в текущем магазине
    products_in_site = set(site_filtered['PRODUCT'].unique())

    # Обновление множества товаров, продающихся во всех магазинах
    if product_presence is None:
        product_presence = products_in_site
    else:
        product_presence &= products_in_site

    return product_presence
###############################################################################################


