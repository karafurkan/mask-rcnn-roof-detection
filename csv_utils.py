"""Handling csv."""
import pandas as pd


def info(file_path: str = "./comparison_all.csv"):
    df = pd.read_csv(
        file_path,
        index_col=0,
    )
    # print(df.head())
    # df = df.tail(300)
    # print(df[0])
    good_count = df["category"].value_counts()["good"]
    bad_count = df["category"].value_counts()["bad"]
    annotate_count = df["category"].value_counts()["annotate"]
    image_count = len(df["category"])
    print(f"Checked images: {image_count}")
    print(f"Good count: {good_count} ({good_count / image_count * 100:.1f}%)")
    print(f"Bad count: {bad_count} ({bad_count / image_count * 100:.1f}%)")
    print(
        f"Annotate count: {annotate_count} ({annotate_count / image_count * 100:.1f}%)"
    )

    # changing paths
    # for i, row in df.iterrows():
    #     p = row["image_path"]
    #     df.loc[i, "image_path"] = p[(p[: p.rfind("/")].rfind("/")) :]

    # df.to_csv("/home/marvin/Desktop/comparison_all.csv")


def concatenate_csv(
    path_start: str,
    file_path_1: str,
    file_path_2: str = "comparison_all.csv",
):
    """Appending csv1 on top of csv2."""
    df1 = pd.read_csv(
        path_start + file_path_1,
        index_col=0,
    )
    df2 = pd.read_csv(
        path_start + file_path_2,
        index_col=0,
    )
    df = df2.append(df1, ignore_index=True)
    print(path_start + "comparison_all.csv")
    df.to_csv(path_start + "comparison_all.csv")


def main():
    path_start = "/home/furkan/gd/pranet/datasets/202109_nrw_cleaned/quality_reviewed/"
    concatenate_csv(
        path_start,
        file_path_1="comparison_is_this_a_good_image?_2023-03-14_16:13:29.csv",
    )
    # info(
    #     "/home/furkan/gd/pranet/datasets/202109_nrw_cleaned/quality_reviewed/comparison_all.csv"
    # )


if __name__ == "__main__":
    main()
