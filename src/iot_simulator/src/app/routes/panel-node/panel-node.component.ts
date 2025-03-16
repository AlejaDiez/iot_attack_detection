import { Component, OnInit } from "@angular/core";
import { ActivatedRoute, RouterModule } from "@angular/router";
import { Node, NodeType } from "@models/node";
import { NgIcon, provideIcons } from "@ng-icons/core";
import { lucideGhost, lucideX } from "@ng-icons/lucide";
import { NetworkManagerService } from "@services/network-manager.service";
import { HlmButtonModule } from "@spartan-ng/ui-button-helm";
import { HlmCardDirective, HlmCardModule } from "@spartan-ng/ui-card-helm";
import { map } from "rxjs";

@Component({
    selector: "app-panel-node",
    imports: [HlmButtonModule, HlmCardModule, NgIcon, RouterModule],
    providers: [provideIcons({ lucideGhost, lucideX })],
    host: {
        class: "fixed right-5 top-1/2 max-h-[calc(100%-2.5rem)] min-w-[400px] max-w-[calc(100%-2.5rem)] -translate-y-1/2 rounded-xl",
    },
    hostDirectives: [HlmCardDirective],
    templateUrl: "panel-node.component.html",
})
export class PanelNodeComponent implements OnInit {
    private _node!: Node;
    protected get ip(): string | undefined {
        return this._node.ip;
    }
    protected get name(): string {
        return this._node.name;
    }
    protected get type(): NodeType {
        return this._node.type;
    }

    public constructor(
        private readonly _route: ActivatedRoute,
        private readonly _networkManager: NetworkManagerService,
    ) {}

    public ngOnInit(): void {
        this._route
            .parent!.params.pipe(
                map(({ mac }) => this._networkManager.findByMac(mac)),
            )
            .subscribe((node) => (this._node = node));
    }
}
