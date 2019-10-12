import argparse
import csv
import scapy.layers.dns
import scapy.sendrecv
import math

from typing import Sequence


def writeto(writer, label):
    csvwriter = csv.writer(writer)
    csvwriter.writerow(["label", "qname"])

    def _prn(pkt):
        dns = pkt.lastlayer()
        if dns.qr == 0:
            csvwriter.writerow([label, dns.qd.qname.decode(errors="ignore")])
    return _prn


def main():
    parser = argparse.ArgumentParser(description="Parse DNS packets")
    parser.add_argument("pcap", metavar="PCAP", type=str,
                        help="Traffic dump.")
    parser.add_argument("out", metavar="OUT", type=str,
                        help="Output attributes.")
    parser.add_argument("label", metavar="label", type=int,
                        help="The class label.")

    args = parser.parse_args()

    # with open(args.out, "w", newline="") as csvfile:
    with open(args.out, "w+") as csvfile:
        scapy.sendrecv.sniff(
            offline=args.pcap,
            store=False,
            lfilter=lambda p: p.haslayer(scapy.layers.dns.DNS),
            prn=writeto(csvfile, args.label))


if __name__ == "__main__":
    main()
