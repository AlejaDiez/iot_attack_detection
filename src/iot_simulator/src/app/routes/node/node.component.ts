import { Component, OnInit } from "@angular/core";
import { ActivatedRoute, RouterModule } from "@angular/router";
import { Node } from "@models/node";
import { NgIcon, provideIcons } from "@ng-icons/core";
import { lucideX } from "@ng-icons/lucide";
import { NetworkManagerService } from "@services/network-manager.service";
import { HlmButtonModule } from "@spartan-ng/ui-button-helm";
import { HlmCardModule } from "@spartan-ng/ui-card-helm";
import { map } from "rxjs";

@Component({
    imports: [HlmButtonModule, HlmCardModule, NgIcon, RouterModule],
    providers: [provideIcons({ lucideX })],
    templateUrl: "node.component.html",
    styles: [
        `
            .animate-slide {
                animation-duration: 400ms;
                animation-timing-function: ease;
                animation-name: slide;
            }

            @keyframes slide {
                from {
                    transform: translate(100%, -50%);
                    opacity: 0;
                }
                to {
                    transform: translate(0, -50%);
                    opacity: 1;
                }
            }
        `,
    ],
})
export class NodeComponent implements OnInit {
    protected node!: Node;

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
