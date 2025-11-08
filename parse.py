import pandas as pd
import re

logfiles = [
    "Attack_free_dataset.txt",
]

for logfile in logfiles:
    timestamps, ids, flags, dlcs, data_bytes = [], [], [], [], []
    try:
        with open(f"datasets/{logfile}", "r") as f:
            for line in f:
                try:
                    match = re.match(
                        r"Timestamp:\s+([\d.]+)\s+ID:\s+([0-9A-Fa-f]+)\s+([0-9A-Fa-f]+)\s+DLC:\s+(\d+)\s+(.+)",
                        line.strip(),
                    )
                except Exception as e:
                    print(f"Error processing: {line.strip()}\n{e}")
                if match:
                    ts, can_id, flag, dlc, data = match.groups()
                    data_list = data.split()

                    data_list = data_list[:8] + ["00"] * (8 - len(data_list))
                    timestamps.append(float(ts))
                    ids.append(int(can_id, 16))  # convert hex to int
                    flags.append(int(flag, 16))
                    dlcs.append(int(dlc))
                    data_bytes.append([int(b, 16) for b in data_list])

        df = pd.DataFrame(data_bytes, columns=[f"data_{i}" for i in range(8)])
        df["timestamp"] = timestamps
        df["can_id"] = ids
        df["flags"] = flags
        df["dlc"] = dlcs

        # reorder columns
        df = df[
            ["timestamp", "can_id", "flags", "dlc"] + [f"data_{i}" for i in range(8)]
        ]
        df.to_csv(f"data/{logfile.split('.')[0]}.csv", index=False)
    except FileNotFoundError:
        print(f"File not found: data/{logfile}")
