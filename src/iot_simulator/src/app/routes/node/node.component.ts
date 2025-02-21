import { Component, OnInit } from "@angular/core";
import { ActivatedRoute, RouterModule } from "@angular/router";
import { NgIcon, provideIcons } from "@ng-icons/core";
import { lucideX } from "@ng-icons/lucide";
import { HlmButtonModule } from "@spartan-ng/ui-button-helm";
import { HlmCardModule } from "@spartan-ng/ui-card-helm";
import { Device } from "../../models/device";
import { Router } from "../../models/router";

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
    protected node!: Device;

    public constructor(private _route: ActivatedRoute) {}

    public ngOnInit(): void {
        this._route.params.subscribe(({ id }) => {
            if (id === "192.168.0.1") {
                this.node = new Router(id, "Router");
            } else {
                this.node = new Device(id, "Dispositivo");
            }
        });
    }
}
