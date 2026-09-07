import sqlite3
import pandas as pd


DB_PATH = "data_pipeline/zepto_books.db"


def run_query(conn, query, query_name):
    print("\n" + "=" * 70)
    print(query_name)
    print("=" * 70)

    print("SQL:")
    print(query)

    result = pd.read_sql_query(
        query,
        conn
    )

    print("\nOutput:")
    print(
        result.to_string(
            index=False
        )
    )

    return result


def main():

    conn = sqlite3.connect(
        DB_PATH
    )

    # =====================================================
    # QUERY 1: SELECT + WHERE
    # =====================================================

    query_1 = """
    SELECT title, price_gbp, rating
    FROM books
    WHERE rating >= 4;
    """

    result_1 = run_query(
        conn,
        query_1,
        "Query 1 - Books with rating >= 4"
    )


    # =====================================================
    # QUERY 2: ORDER BY + LIMIT
    # =====================================================

    query_2 = """
    SELECT title, price_inr
    FROM books
    ORDER BY price_inr DESC
    LIMIT 10;
    """

    result_2 = run_query(
        conn,
        query_2,
        "Query 2 - 10 Most Expensive Books"
    )


    # =====================================================
    # QUERY 3: DISTINCT
    # =====================================================

    query_3 = """
    SELECT DISTINCT category_name
    FROM categories;
    """

    result_3 = run_query(
        conn,
        query_3,
        "Query 3 - Distinct Categories"
    )


    # =====================================================
    # QUERY 4: IN + JOIN
    # =====================================================

    query_4 = """
    SELECT
        b.title,
        c.category_name,
        b.rating
    FROM books b
    INNER JOIN categories c
        ON b.category_id = c.category_id
    WHERE c.category_name IN ('Travel', 'Mystery')
    ORDER BY b.rating DESC;
    """

    result_4 = run_query(
        conn,
        query_4,
        "Query 4 - Books from Travel or Mystery"
    )


    # =====================================================
    # QUERY 5: BETWEEN
    # =====================================================

    query_5 = """
    SELECT title, price_gbp
    FROM books
    WHERE price_gbp BETWEEN 20 AND 40
    ORDER BY price_gbp;
    """

    result_5 = run_query(
        conn,
        query_5,
        "Query 5 - Books priced between GBP 20 and 40"
    )


    # =====================================================
    # QUERY 6: JOIN
    # =====================================================

    query_6 = """
    SELECT
        b.title,
        b.rating,
        b.price_gbp,
        c.category_name
    FROM books b
    INNER JOIN categories c
        ON b.category_id = c.category_id
    ORDER BY
        b.rating DESC,
        c.category_name,
        b.title;
    """

    result_6 = run_query(
        conn,
        query_6,
        "Query 6 - Books joined with Categories"
    )


    # =====================================================
    # TASK 6A: TWO pd.read_sql_query RESULTS
    # =====================================================

    print("\n" + "=" * 70)
    print("TASK 6A - pd.read_sql_query RESULTS")
    print("=" * 70)


    sql_df_1 = pd.read_sql_query(
        """
        SELECT
            title,
            price_gbp,
            rating
        FROM books
        WHERE rating >= 4
        ORDER BY rating DESC;
        """,
        conn
    )


    sql_df_2 = pd.read_sql_query(
        """
        SELECT
            b.title,
            b.rating,
            b.price_gbp,
            c.category_name
        FROM books b
        INNER JOIN categories c
            ON b.category_id = c.category_id
        ORDER BY
            b.rating DESC,
            c.category_name,
            b.title;
        """,
        conn
    )


    print(
        "\nResult from pd.read_sql_query - Query 1:"
    )

    print(
        sql_df_1.to_string(
            index=False
        )
    )


    print(
        "\nResult from pd.read_sql_query - JOIN Query:"
    )

    print(
        sql_df_2.to_string(
            index=False
        )
    )


    # =====================================================
    # TASK 6B: REPRODUCE JOIN USING pd.merge()
    # =====================================================

    books_df = pd.read_sql_query(
        """
        SELECT
            book_id,
            title,
            price_gbp,
            rating,
            category_id
        FROM books;
        """,
        conn
    )


    categories_df = pd.read_sql_query(
        """
        SELECT
            category_id,
            category_name
        FROM categories;
        """,
        conn
    )


    merge_df = pd.merge(
        books_df,
        categories_df,
        on="category_id",
        how="inner"
    )


    merge_df = merge_df[
        [
            "title",
            "rating",
            "price_gbp",
            "category_name"
        ]
    ]


    merge_df = (
        merge_df
        .sort_values(
            by=[
                "rating",
                "category_name",
                "title"
            ],
            ascending=[
                False,
                True,
                True
            ]
        )
        .reset_index(
            drop=True
        )
    )


    sql_join_sorted = (
        sql_df_2[
            [
                "title",
                "rating",
                "price_gbp",
                "category_name"
            ]
        ]
        .sort_values(
            by=[
                "rating",
                "category_name",
                "title"
            ],
            ascending=[
                False,
                True,
                True
            ]
        )
        .reset_index(
            drop=True
        )
    )


    print("\n" + "=" * 70)
    print("TASK 6B - SQL JOIN VS pd.merge()")
    print("=" * 70)


    print(
        "\nSQL JOIN result:"
    )

    print(
        sql_join_sorted.to_string(
            index=False
        )
    )


    print(
        "\npd.merge() result:"
    )

    print(
        merge_df.to_string(
            index=False
        )
    )


    equivalent = (
        sql_join_sorted.equals(
            merge_df
        )
    )


    print(
        "\nAre SQL JOIN and pd.merge() equivalent?",
        equivalent
    )


    # =====================================================
    # SAVE OUTPUTS
    # =====================================================

    result_1.to_csv(
        "data_pipeline/query_1_output.csv",
        index=False
    )

    result_2.to_csv(
        "data_pipeline/query_2_output.csv",
        index=False
    )

    result_3.to_csv(
        "data_pipeline/query_3_output.csv",
        index=False
    )

    result_4.to_csv(
        "data_pipeline/query_4_output.csv",
        index=False
    )

    result_5.to_csv(
        "data_pipeline/query_5_output.csv",
        index=False
    )

    result_6.to_csv(
        "data_pipeline/query_6_join_output.csv",
        index=False
    )

    sql_df_1.to_csv(
        "data_pipeline/pd_read_sql_output.csv",
        index=False
    )

    merge_df.to_csv(
        "data_pipeline/pd_merge_output.csv",
        index=False
    )


    conn.close()


    print("\n" + "=" * 70)
    print("ALL QUERY OUTPUTS SAVED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()