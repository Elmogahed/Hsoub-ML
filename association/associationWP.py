import mysql.connector

connection_mydb = mysql.connector.connect(
    host = 'localhost',
    user = 'root',
    password = '',
    database = 'wp-ecommerce'
)

def get_product_name_from_id(connection_mydb,product_id):
    cursor = connection_mydb.cursor(dictionary=True)
    sql = "SELECT post_title FROM wp_posts WHERE ID=(%s)"
    id = (product_id,)
    cursor.execute(sql,id)
    results = cursor.fetchall()
    if len(results) > 0:
        return results[0]['post_title']
    return "Unknown Product"
get_product_name_from_id(connection_mydb, 55945)
get_product_name_from_id(connection_mydb, 99999)

import pandas as pd
def build_dataframe_associated_products(connection_mydb):
    df = pd.DataFrame(columns=[0,1,2,3,4,5,6,7,8,9])
    cursor = connection_mydb.cursor(dictionary=True)
    sql = "SELECT * FROM wp_wc_order_stats order by order_id "
    cursor.execute(sql)
    results_orders = cursor.fetchall()
    for order in results_orders:
        order_id  = order['order_id']
        sql = "SELECT * FROM wp_wc_order_product_lookup where order_id=(%s)"
        id = (order_id,)
        cursor.execute(sql,id)
        results_products = cursor.fetchall()
        products_ids = []
        for product in results_products:
            products_id = product['product_id']
            if products_id > 0:
                products_ids.append(products_id)
            if len(products_ids) > 1:
                df = pd.concat([df, pd.DataFrame([products_id])], ignore_index=True)
    return df

df = build_dataframe_associated_products(connection_mydb)
print(df)