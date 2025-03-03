import { Component, OnInit } from "@angular/core";
import { ActivatedRoute, RouterModule } from "@angular/router";
import { Node } from "@models/node";
import { Router } from "@models/router";
import { NgIcon, provideIcons } from "@ng-icons/core";
import { lucideX } from "@ng-icons/lucide";
import { HlmButtonModule } from "@spartan-ng/ui-button-helm";
import { HlmCardModule } from "@spartan-ng/ui-card-helm";

@Component({
    imports: [HlmButtonModule, HlmCardModule, NgIcon, RouterModule],
    providers: [provideIcons({ lucideX })],
    templateUrl: "node.component.html",
    styles: [
        `
            .animate-slide {
                animation-name: slide;
                animation-duration: 400ms;
                animation-timing-function: ease;
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

    public constructor(private _route: ActivatedRoute) {}

    public ngOnInit(): void {
        this.node = new Router("Router");
        this._route.parent!.params.subscribe(({ mac }) => {
            // Get node from network manager service
        });
    }
}
