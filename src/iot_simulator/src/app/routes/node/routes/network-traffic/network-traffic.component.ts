import { Component, OnInit } from "@angular/core";
import { ActivatedRoute } from "@angular/router";
import { NgIcon, provideIcons } from "@ng-icons/core";
import { lucideForward, lucideRepeat, lucideReply } from "@ng-icons/lucide";
import { BrnSelectModule } from "@spartan-ng/brain/select";
import { BrnTableModule } from "@spartan-ng/brain/table";
import { HlmButtonModule } from "@spartan-ng/ui-button-helm";
import { HlmInputModule } from "@spartan-ng/ui-input-helm";
import { HlmSelectModule } from "@spartan-ng/ui-select-helm";
import { HlmTableModule } from "@spartan-ng/ui-table-helm";
import { HlmTabsModule } from "@spartan-ng/ui-tabs-helm";
import { Device } from "../../../../models/device";
import { Router } from "../../../../models/router";

@Component({
    imports: [
        BrnSelectModule,
        BrnTableModule,
        HlmButtonModule,
        HlmInputModule,
        HlmSelectModule,
        HlmTableModule,
        HlmTabsModule,
        NgIcon,
    ],
    providers: [provideIcons({ lucideForward, lucideRepeat, lucideReply })],
    templateUrl: "network-traffic.component.html",
})
export class NetworkTrafficComponent implements OnInit {
    protected node!: Device;
    protected networkTraces: {
        id: number;
        source: string;
        destination: string;
        length: number;
        data: string;
    }[] = [
        {
            id: 1,
            source: "192.168.0.31",
            destination: "192.168.0.24",
            length: 512,
            data: "Hello from 31 to 24",
        },
        {
            id: 2,
            source: "192.168.0.31",
            destination: "192.168.0.1",
            length: 256,
            data: "Routing request from 31",
        },
        {
            id: 3,
            source: "192.168.0.1",
            destination: "192.168.0.24",
            length: 256,
            data: "Forwarding from 31",
        },
        {
            id: 4,
            source: "192.168.0.24",
            destination: "192.168.0.31",
            length: 512,
            data: "Reply from 24 to 31",
        },
        {
            id: 5,
            source: "192.168.0.24",
            destination: "192.168.0.1",
            length: 128,
            data: "Requesting status",
        },
        {
            id: 6,
            source: "192.168.0.1",
            destination: "192.168.0.24",
            length: 128,
            data: "Status OK",
        },
        {
            id: 7,
            source: "192.168.0.31",
            destination: "192.168.0.1",
            length: 300,
            data: "Sending file metadata",
        },
        {
            id: 8,
            source: "192.168.0.1",
            destination: "192.168.0.24",
            length: 300,
            data: "Forwarding file metadata",
        },
        {
            id: 9,
            source: "192.168.0.24",
            destination: "192.168.0.1",
            length: 1024,
            data: "Uploading file data",
        },
        {
            id: 10,
            source: "192.168.0.1",
            destination: "192.168.0.31",
            length: 1024,
            data: "Forwarding file data",
        },
        {
            id: 11,
            source: "192.168.0.31",
            destination: "192.168.0.24",
            length: 200,
            data: "Sending acknowledgment",
        },
        {
            id: 12,
            source: "192.168.0.24",
            destination: "192.168.0.31",
            length: 200,
            data: "Acknowledgment received",
        },
    ];

    public constructor(private _route: ActivatedRoute) {}

    public ngOnInit(): void {
        this._route.parent!.params.subscribe(({ id }) => {
            if (id === "192.168.0.1") {
                this.node = new Router(id, "Router");
            } else {
                this.node = new Device(id, "Dispositivo");
            }
        });
    }
}
