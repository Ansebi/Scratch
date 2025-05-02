from general import *
from distribution_analysis import *
import dataloader


def slice_quarter(df_sales_product, quarter):
    df_quarter = df_sales_product.copy()
    df_quarter['quarter'] = df_quarter['DATE'].dt.year.astype(str)\
        + 'q'\
        + df_quarter['DATE'].dt.quarter.astype(str)
    df_quarter = df_quarter\
        .query('quarter == @quarter')\
        .drop(columns='quarter')\
        .reset_index(drop=True)
    return df_quarter


def get_product_name(df_products, product):
    try:
        condition = df_products['artcexr'] == product
        product_name = df_products[condition]\
            .iloc[0]['name_art']
    except IndexError:
        product_name = ''
    return product_name


def get_site_name(df_sites, site):
    try:
        condition = df_sites['id_sym'] == site
        site_name = df_sites[condition]\
            .iloc[0]['name']
    except IndexError:
        site_name = ''
    return site_name


def collect_distribution_metrics(
    df_assort,
    products,
    quarter: str = None,
    write_parquet: bool = True
):
    '''
    quarter (str): e.g. '2023q3'
    '''
    distribution_metrics = []
    for product in tqdm.tqdm(products):
        df_sales_product = dataloader.product_load_sales(product)
        if quarter is not None:
            df_sales_product = slice_quarter(df_sales_product, quarter)
        product_sites = df_assort.loc[product, 'id_sym']
        for site in product_sites:
            df_sales_site = df_sales_product.query('SITE == @site')
            if df_sales_site.empty:
                continue
            agg = {
                'SALES_QUANTITY': 'sum'
            }
            df_sales_site = df_sales_site\
                .groupby('DATE')\
                .agg(agg)\
                .reset_index()
            df_sales_site = recover_timeline(df_sales_site, 'DATE')\
                .fillna(0)
            sales_quantity = df_sales_site['SALES_QUANTITY'].to_numpy()
            site_metrics = compute_distribution_metrics(sales_quantity) 
            entry = {
                    'PRODUCT': product,
                    'SITE': site
            }        
            entry.update(site_metrics)
            distribution_metrics.append(entry)
    df_distribution_metrics = pd.DataFrame(distribution_metrics)
    if write_parquet:
        date = pd.Timestamp('now').date()
        if quarter is not None:
            distribution_metrics_filename = f'distribution_metrics_{quarter}_{date}.pq'
        else:
            distribution_metrics_filename = f'distribution_metrics_{date}.pq'
        distribution_metrics_path = '/'.join(
            [
                CONFIG['DATA_DIR'],
                'sales',
                distribution_metrics_filename
            ]
        )
        df_distribution_metrics.to_parquet(distribution_metrics_path)
        print(f'Wrote distribution metrics to\n{distribution_metrics_path}')
    return df_distribution_metrics


def plot_sales(df_products, df_sites, product, site, quarter: str = None):
    product_name = get_product_name(df_products, product)
    site_name = get_site_name(df_sites, site)
    df_sales_product = dataloader.product_load_sales(product)
    if quarter is not None:
        df_sales_product = slice_quarter(df_sales_product, quarter)
    df_sales_site = df_sales_product.query('SITE == @site')
    agg = {
        'SALES_QUANTITY': 'sum'
    }
    df_sales_site = df_sales_site\
        .groupby('DATE')\
        .agg(agg)\
        .reset_index()
    df_sales_site = recover_timeline(df_sales_site, 'DATE')\
        .fillna(0)
    sales_quantity = df_sales_site['SALES_QUANTITY'].to_numpy()
    date = df_sales_site['DATE']
    colors = np.where(sales_quantity == 0, 'red', 'green')
    mean_sales = np.mean(sales_quantity)

    plt.figure(figsize=(20, 3))
    plt.scatter(
        date,
        sales_quantity,
        s=5,
        color=colors
    )
    plt.axhline(y=mean_sales, color='grey', linestyle='--', linewidth=1)
    plt.xticks(
        pd.date_range(start=date.min(), end=date.max(), freq='MS'), 
        rotation=60, 
        labels=pd.date_range(start=date.min(), end=date.max(), freq='MS').strftime('%Y-%m')
    )
    plt.xlabel('Date')
    plt.ylabel('Sales Quantity')
    title = f'{product} ({product_name})\n{site} ({site_name})'
    if quarter is not None:
        title = f'{quarter}\n' + title
    plt.title(title)
    plt.show()


def process_selection_quarter(
    df_products: pd.DataFrame,
    df_sites: pd.DataFrame,
    selection: pd.DataFrame,
    selection_name: str,
    quarter: str = None
):
    printeable = selection[['PRODUCT', 'SITE']]\
        .merge(
            df_products[['artcexr', 'name_art']]\
                .drop_duplicates(),
            left_on='PRODUCT',
            right_on='artcexr',
            how='left'
        ).drop(columns='artcexr')\
        .rename(columns={'name_art': 'PRODUCT_NAME'})
    printeable = printeable\
        .merge(
            df_sites[['id_sym', 'name']]\
                .drop_duplicates(),
            left_on='SITE',
            right_on='id_sym',
            how='left'
        ).drop(columns='id_sym')\
        .rename(columns={'name': 'SITE_NAME'})
    printeable['SELECTION'] = selection_name
    columns = [
        'SELECTION',
        'PRODUCT_NAME',
        'PRODUCT',
        'SITE_NAME',
        'SITE'
    ]
    if quarter is not None:
        printeable['QUARTER'] = quarter
        columns = ['QUARTER'] + columns
    printeable = printeable[columns]
    return printeable


def plot_selection_quarter(df_products, df_sites, selection, quarter):
    for _, row in selection.iterrows():
        product = row['PRODUCT']
        site = row['SITE']
        plot_sales(df_products, df_sites, product, site, quarter)


def selection_routine(selections, df_products, df_sites, selection, selection_name, quarter=None):
    selection_printeable = process_selection_quarter(df_products, df_sites, selection, selection_name, quarter)
    IPython.display.display(selection)
    plot_selection_quarter(df_products, df_sites, selection, quarter)
    selections.append(selection_printeable)