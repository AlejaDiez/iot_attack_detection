import { Component, OnInit } from "@angular/core";
import { ActivatedRoute } from "@angular/router";
import { Node } from "@models/node";
import { NgIcon, provideIcons } from "@ng-icons/core";
import { lucideForward, lucideRepeat, lucideReply } from "@ng-icons/lucide";
import { NetworkManagerService } from "@services/network-manager.service";
import { BrnSelectModule } from "@spartan-ng/brain/select";
import { BrnTableModule } from "@spartan-ng/brain/table";
import { HlmButtonModule } from "@spartan-ng/ui-button-helm";
import { HlmInputModule } from "@spartan-ng/ui-input-helm";
import { HlmSelectModule } from "@spartan-ng/ui-select-helm";
import { HlmTableModule } from "@spartan-ng/ui-table-helm";
import { HlmTabsModule } from "@spartan-ng/ui-tabs-helm";
import { map } from "rxjs";

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
    protected node!: Node;
    protected get traffic(): {
        id: number;
        srcIP: string;
        dstIP: string;
        payloadSize: number;
        payload?: string;
    }[] {
        return (
            this.node?.traffic.map(
                ({ srcIP, dstIP, payloadSize, payload }, id) => ({
                    id,
                    srcIP,
                    dstIP,
                    payloadSize,
                    payload,
                }),
            ) ?? []
        );
    }

    public constructor(
        private _route: ActivatedRoute,
        private _networkManager: NetworkManagerService,
    ) {}

    public ngOnInit(): void {
        this._route
            .parent!.params.pipe(
                map(({ mac }) => this._networkManager.findByMac(mac)),
            )
            .subscribe((node) => (this.node = node));
    }
}
